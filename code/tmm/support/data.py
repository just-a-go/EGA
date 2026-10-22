"""Source records, participant splits, and content-preserving renderers."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import time


class ObservationError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def csv_rows(path, delimiter=','):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def native_daic(root, spec):
    result = []
    for split, source in spec['splits'].items():
        ids = (root / source['ids_file']).read_text().split()
        rows = csv_rows(source['label_csv'])
        labels = {}
        for raw in rows:
            r = {k.strip().casefold(): v for k, v in raw.items()}
            id_col = source.get('id_column', spec['id_column']).casefold()
            label_col = source.get('label_column', spec['label_column']).casefold()
            labels[str(int(float(r[id_col])))] = int(r[label_col])
        if len(ids) != source['expected_n'] or len(ids) != len(set(ids)):
            raise ValueError(f'Invalid frozen {split} participant list')
        for sid in ids:
            if sid not in labels or labels[sid] not in (0, 1):
                raise ValueError(f'Missing native binary label: {sid}')
            transcript = Path(spec['transcript_pattern'].format(id=sid))
            audio = spec['audio_pattern'].format(id=sid)
            views = []
            if transcript.is_file():
                for n, row in enumerate(csv_rows(transcript, '\t')):
                    start, stop = float(row['start_time']), float(row['stop_time'])
                    text = row['value'].strip()
                    max_duration = spec.get('max_turn_seconds', 20)
                    if not text or stop <= start or (max_duration is not None and stop-start > max_duration):
                        continue
                    base = {'event': f'turn_{n}', 'start': start, 'stop': stop,
                            'speaker': row['speaker'].strip(), 'version': 'released',
                            'source': str(transcript), 'available': True}
                    views.append(base | {'id': f'turn_{n}:T', 'modality': 'T', 'text': text})
                    views.append(base | {'id': f'turn_{n}:A', 'modality': 'A',
                                         'source': audio, 'path': audio, 'available': Path(audio).is_file()})
            result.append({'id': sid, 'participant': sid, 'family': spec['family'],
                           'split': split, 'label': labels[sid], 'views': views})
    return result


def load_records(root, spec):
    started = time.perf_counter()
    if spec.get('format') == 'daic':
        rows = native_daic(root, spec)
    else:
        path = root / spec['manifest']
        if not path.is_file():
            raise FileNotFoundError(f'Required source manifest: {path}')
        rows = read_jsonl(path)
    validate_records(rows, nested=spec.get('evaluation') == 'nested')
    for rec in rows:
        for view in rec['views']:
            reserve = len(render([view]).encode('utf-8')) + 64
            if view['modality'] == 'A':
                reserve += int(np.ceil((view['stop']-view['start'])*64))
            elif view['modality'] == 'F':
                reserve += 8*len(view['features'])
            view['context_reservation'] = reserve
    elapsed = time.perf_counter()-started
    for row in rows:
        row['index_seconds'] = elapsed/len(rows)
    return rows


def validate_records(rows, nested=False):
    if not rows:
        raise ValueError('Empty dataset')
    ids, groups = set(), {}
    for rec in rows:
        key = rec['id']
        if key in ids or rec['label'] not in (0, 1):
            raise ValueError('Duplicate sample or invalid native label')
        ids.add(key)
        group = (rec['family'], rec['participant'])
        partition = rec.get('split', 'pool')
        if group in groups and groups[group] != partition:
            raise ValueError(f'Participant crosses partitions: {group}')
        groups[group] = partition
        if not nested and partition not in ('train', 'dev', 'test', 'unused'):
            raise ValueError(f'Invalid partition: {partition}')
        seen = set()
        for view in rec['views']:
            if view['id'] in seen or not 0 <= view['start'] < view['stop']:
                raise ValueError('Duplicate observation or invalid source interval')
            seen.add(view['id'])
            if view['modality'] not in ('T', 'A', 'F') or not view.get('event'):
                raise ValueError('Invalid modality or missing event identity')
            if not view.get('source') or not view.get('version'):
                raise ValueError('Missing source provenance/version')


def canonical(views):
    unique = {}
    for view in views:
        if view['id'] in unique and unique[view['id']] != view:
            raise ValueError('Observation identity collision')
        unique[view['id']] = view
    return sorted(unique.values(), key=lambda v: (v['start'], v['modality'], v['id']))


def pool(rec, track):
    modalities = {'T': {'T'}, 'A': {'A'}, 'A+T': {'A', 'T'}, 'F': {'F'}}[track]
    return canonical([v for v in rec['views'] if v['modality'] in modalities and v.get('available', True)])


def seed_views(rec, track, panel=False):
    candidates = pool(rec, track)
    kind = 'T' if 'T' in track else track
    modality_candidates = [v for v in candidates if v['modality'] == kind]
    candidates = [v for v in modality_candidates
                  if kind in ('F', 'A') or v.get('speaker', '').lower() == 'participant']
    if len(candidates) < 3 and not any(v.get('speaker', '').lower() == 'participant' for v in modality_candidates):
        candidates = [v for v in modality_candidates if v.get('speaker', 'unknown').lower() == 'unknown']
    if len(candidates) < 3:
        raise ObservationError('Fewer than three eligible seed observations')
    lo, hi = candidates[0]['start'], candidates[-1]['stop']
    chosen = []
    for f in (1/6, 1/2, 5/6):
        chosen.append(min((v for v in candidates if v not in chosen),
                          key=lambda v: (abs((v['start'] + v['stop']) / 2 - lo - f * (hi-lo)), v['id'])))
    if panel and track == 'A+T':
        for text in list(chosen):
            match = [v for v in pool(rec, track) if v['modality'] == 'A' and v['event'] == text['event']]
            if len(match) != 1:
                raise ObservationError('Seed audio unavailable or ambiguous')
            chosen.extend(match)
    return canonical(chosen)


def render(views, renderer=0, repeat_id=None):
    items = canonical(views)
    if renderer == 3:
        # Acquisition and its repeated-source control occupy the same final slot.
        items = [v for v in items if v['id'] != repeat_id] + [v for v in items if v['id'] == repeat_id]
    blocks = []
    for v in items:
        unit = v.get('coordinate_unit', 'seconds')
        interval = f"{v['start']:.3f}--{v['stop']:.3f}" + ('s' if unit == 'seconds' else f' {unit}')
        meta = f"[{v['id']}] {interval} | {v.get('speaker', 'unknown')} | {v['modality']}"
        payload = v.get('text', f"<{v['modality']} observation>")
        if v['modality'] == 'F':
            payload += ' Available feature modalities: ' + ', '.join(sorted(v['features']))
        blocks.append((v['id'], f'{payload}\n{meta}' if renderer == 1 else f'{meta}\n{payload}'))
    if renderer == 2 and items:
        repeat_id = repeat_id or items[0]['id']
        match = next(text for key, text in blocks if key == repeat_id)
        blocks.append((repeat_id, 'Repeated source view; no new observation:\n' + match))
    return '\n\n'.join(text for _, text in blocks)


def metadata(view, state):
    kind = [float(view['modality'] == m) for m in ('T', 'A', 'F')]
    shared = any(v['event'] == view['event'] for v in state)
    coordinate_scale = (3600, 20) if view.get('coordinate_unit', 'seconds') in ('second', 'seconds') else (100, 1)
    return np.array(kind + [view['start']/coordinate_scale[0], (view['stop']-view['start'])/coordinate_scale[1],
                           float(view.get('available', True)), float(shared)], dtype=np.float32)


def participant_folds(rows, count, seed):
    from sklearn.model_selection import StratifiedKFold
    groups = {}
    for row in rows:
        key = (row['family'], row['participant'])
        if key in groups and groups[key] != row['label']:
            raise ValueError('Participant has inconsistent labels')
        groups[key] = row['label']
    keys = sorted(groups)
    labels = [groups[k] for k in keys]
    if min(labels.count(0), labels.count(1)) < count:
        raise ValueError('Too few participants per class for declared folds')
    splitter = StratifiedKFold(count, shuffle=True, random_state=seed)
    for train, test in splitter.split(keys, labels):
        train_keys, test_keys = {keys[i] for i in train}, {keys[i] for i in test}
        yield ([r for r in rows if (r['family'], r['participant']) in train_keys],
               [r for r in rows if (r['family'], r['participant']) in test_keys])
