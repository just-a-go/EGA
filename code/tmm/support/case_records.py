"""Post-hoc EGA case records from saved predictions; no model or training imports."""
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items() if k != 'controller_context'}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, path)


def _read_rows(path):
    if not Path(path).is_file():
        return {}
    return {row['case_id']: row for row in (
        json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip())}


def _write_rows(path, rows):
    _atomic(path, ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n'
                         for row in rows))


def _key(row, metadata):
    fold = row.get('outer_fold', metadata.get('outer_fold'))
    return str(row['family']), str(row['id']), fold


def _valid(row):
    p = row.get('p')
    return (row.get('status') == 'ok' and row.get('probability_valid', True)
            and isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1)


def _observed_ids(prediction):
    context = prediction.get('controller_context') or {}
    return (context.get('observed_view_ids') or prediction.get('state_reader_observed_view_ids')
            or prediction.get('observed_view_ids'))


def _seed_ids(record, track):
    """Fallback for historical predictions lacking explicit observed identities."""
    if track not in ('T', 'A', 'A+T', 'F'):
        return []
    kind = 'T' if 'T' in track else track
    candidates = [v for v in record.get('views', []) if v['modality'] == kind and v.get('available', True)]
    if kind == 'T':
        participant = [v for v in candidates if v.get('speaker', '').lower() == 'participant']
        candidates = participant or [v for v in candidates if v.get('speaker', 'unknown').lower() == 'unknown']
    candidates.sort(key=lambda v: (v['start'], v['modality'], v['id']))
    if len(candidates) < 3:
        return []
    lo, hi, chosen = candidates[0]['start'], candidates[-1]['stop'], []
    for fraction in (1/6, 1/2, 5/6):
        chosen.append(min((v for v in candidates if v not in chosen),
            key=lambda v: (abs((v['start']+v['stop'])/2-lo-fraction*(hi-lo)), v['id'])))
    return [v['id'] for v in sorted(chosen, key=lambda v: (v['start'], v['modality'], v['id']))]


def _view(view):
    keys = ('id', 'event', 'modality', 'speaker', 'start', 'stop', 'coordinate_unit',
            'text', 'source', 'path', 'audio_path', 'available', 'features')
    return _clean({key: view[key] for key in keys if key in view})


def _prediction(row, cut):
    value = _clean(row)
    value['prediction_class'] = int(row['p'] >= cut) if _valid(row) else None
    value['raw_p'] = _clean(row.get('raw_p'))
    value['valid_for_case_classification'] = bool(_valid(row))
    return value


def _case(before, after, record, fingerprint, metadata):
    family, identity, fold = _key(before, metadata)
    if any(str(row.get('participant', record['participant'])) != str(record['participant'])
           for row in (before, after)):
        raise ValueError('Case record participant differs from the evaluation manifest')
    if (int(before['label']) != int(after['label']) or int(record['label']) != int(before['label'])
            or int(before['label']) not in (0, 1)):
        raise ValueError('Case record requires the same native binary label in manifest/Seed/EGA')
    cut = before.get('threshold', metadata.get('threshold'))
    if after.get('threshold', cut) != cut or not isinstance(cut, (int, float)) or not math.isfinite(cut):
        raise ValueError('Case record requires one saved shared classifier threshold')
    valid = _valid(before) and _valid(after)
    label = int(before['label'])
    seed_class = int(before['p'] >= cut) if valid else None
    ega_class = int(after['p'] >= cut) if valid else None
    outcome = ('correction' if seed_class != label and ega_class == label else
               'harm' if seed_class == label and ega_class != label else
               'stable_correct' if ega_class == label else 'stable_wrong') if valid else 'invalid_prediction'
    delta = (after['p']-label)**2-(before['p']-label)**2 if valid else None
    seed_ids = list(_observed_ids(before) or _seed_ids(record, metadata.get('track')))
    selected = [step.get('view', step.get('action_id')) for step in after.get('trace', [])]
    final_ids = list(_observed_ids(after) or dict.fromkeys(seed_ids+[v for v in selected if v]))
    views = {v['id']: v for v in record.get('views', [])}
    case_id = hashlib.sha256(json.dumps([fingerprint, family, identity, fold],
                                        separators=(',', ':')).encode()).hexdigest()
    return {'schema': 'ega-case-v1', 'case_id': case_id, 'evaluation_fingerprint': fingerprint,
        'family': family, 'id': identity, 'participant': record['participant'], 'outer_fold': fold,
        'label': label, 'threshold': cut, 'outcome': outcome,
        'probability_improved': delta is not None and delta < 0,
        'probability_improvement_without_class_flip': bool(valid and seed_class == ega_class and delta < 0),
        'delta_brier': delta, 'seed': _prediction(before, cut), 'ega': _prediction(after, cut),
        'observed': {'seed_view_ids': seed_ids, 'ega_view_ids': final_ids,
            'seed_views': [_view(views[v]) for v in seed_ids if v in views],
            'ega_views': [_view(views[v]) for v in final_ids if v in views],
            'missing_view_ids': [v for v in dict.fromkeys(seed_ids+final_ids) if v not in views],
            'identity_basis': 'saved_prediction_ids_or_seed_plus_selected_trace'},
        'run_metadata': _clean(metadata),
        'interpretation': 'Observed prediction transition only; no causal explanation is inferred.'}


def export_case_records(predictions, records, out, run_metadata=None, index_path=None):
    """Upsert saved pairs. index_path is the project index directory, one atomic fragment per run."""
    metadata = dict(run_metadata or {})
    fingerprint = predictions.get('fingerprint', metadata.get('evaluation_fingerprint'))
    if not fingerprint:
        raise ValueError('Case export requires the saved evaluation fingerprint')
    metadata.setdefault('threshold', predictions.get('threshold'))
    metadata.setdefault('run_directory', str(Path(out)))
    seed = {_key(row, metadata): row for row in predictions['seed']}
    ega = {_key(row, metadata): row for row in predictions['ega']}
    if len(seed) != len(predictions['seed']) or len(ega) != len(predictions['ega']) or seed.keys() != ega.keys():
        raise ValueError('Case export requires unique matching family/id/outer-fold prediction pairs')
    lookup = {(str(r['family']), str(r['id'])): r for r in records}
    if len(lookup) != len(records):
        raise ValueError('Case manifest contains duplicate family/id records')
    directory = Path(out)/'cases'
    path = directory/'all_cases.jsonl'
    cases = _read_rows(path)
    if any(row['evaluation_fingerprint'] != fingerprint for row in cases.values()):
        raise ValueError('Run case directory belongs to another evaluation fingerprint')
    for key, before in seed.items():
        record = lookup.get(key[:2])
        if record is None:
            raise ValueError('Case prediction has no matching evaluation manifest record: '+str(key))
        case = _case(before, ega[key], record, fingerprint, metadata)
        cases[case['case_id']] = case
    ordered = sorted(cases.values(), key=lambda row: (str(row['outer_fold']), row['family'], row['id']))
    _write_rows(path, ordered)
    for filename, predicate in (
            ('corrections.jsonl', lambda row: row['outcome'] == 'correction'),
            ('harms.jsonl', lambda row: row['outcome'] == 'harm'),
            ('probability_improvements.jsonl', lambda row: row['probability_improvement_without_class_flip'])):
        _write_rows(directory/filename, [row for row in ordered if predicate(row)])
    counts = dict(Counter(row['outcome'] for row in ordered))
    summary = {'schema': 'ega-case-manifest-v1', 'evaluation_fingerprint': fingerprint,
        'cases': len(cases), 'outcomes': counts, 'probability_improvements_without_flip':
        sum(row['probability_improvement_without_class_flip'] for row in ordered),
        'files': ['all_cases.jsonl', 'corrections.jsonl', 'harms.jsonl', 'probability_improvements.jsonl']}
    _atomic(directory/'manifest.json', json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    if index_path is not None:
        index_path = Path(index_path)
        # Accept the initial index.jsonl API without writing a shared global file.
        if index_path.suffix == '.jsonl':
            index_path = index_path.with_suffix('')
        fragment = hashlib.sha256(str(Path(out).resolve()).encode()).hexdigest()[:16]
        index_file = index_path/fingerprint/(fragment+'.jsonl')
        index = []
        for row in ordered:
            entry = {key: row[key] for key in (
                'case_id', 'evaluation_fingerprint', 'family', 'id', 'participant', 'outer_fold',
                'label', 'threshold', 'outcome', 'delta_brier', 'probability_improvement_without_class_flip')}
            entry.update(cases_file=str(path.resolve()), run_directory=str(Path(out).resolve()))
            index.append(entry)
        _write_rows(index_file, index)
        summary['index_file'] = str(index_file)
    return summary
