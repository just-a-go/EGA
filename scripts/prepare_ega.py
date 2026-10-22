"""Build source-backed EGA inputs and freeze documented participant partitions.

No model generation, training, data downloads, or clinical annotations are created.
Run on the data host; retain final annotation content and participant partitions.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


DATASETS = {'daic': 'DAIC-WOZ', 'ptsd': 'E-DAIC', 'eatd': 'EATD',
            'mmpsy_depression': 'MMPsy', 'mmpsy_anxiety': 'MMPsy',
            'dvlog': 'D-Vlog', 'modma': 'MODMA'}


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def dump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def csv_lookup(path, id_column):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return {str(int(float(row[id_column]))): row for row in csv.DictReader(handle)}


def label_map(root, data, task):
    if task == 'daic':
        lookup = {}
        for split, filename, column in [('train', 'train_split_Depression_AVEC2017.csv', 'PHQ8_Binary'),
                                         ('dev', 'dev_split_Depression_AVEC2017.csv', 'PHQ8_Binary'),
                                         ('test', 'full_test_split.csv', 'PHQ_Binary')]:
            path = data / 'DAIC-WOZ' / filename
            for sid, row in csv_lookup(path, 'Participant_ID').items():
                lookup[sid] = (int(row[column]), {'path': str(path), 'column': column})
        return lookup
    if task == 'ptsd':
        path = data / 'E-DAIC/metadata_mapped.csv'
        return {sid: (int(row['PCL-C (PTSD)']), {'path': str(path), 'column': 'PCL-C (PTSD)'})
                for sid, row in csv_lookup(path, 'Participant_ID').items()}
    if task == 'eatd':
        path = root / 'manifests/sources/eatd_verified_labels.json'
        rows = json.loads(path.read_text(encoding='utf-8'))
        return {row['participant']: (row['label'], row['provenance']) for row in rows}
    if task.startswith('mmpsy_'):
        path = root / 'annotations/evidence/official_sources/mmpsy_wav_select.json'
        dimension = 'phq-label' if task.endswith('depression') else 'gad-label'
        return {row['user_id']: (int(float(row[dimension]['score']) >= 10),
                {'path': str(path), 'field': dimension + '.score', 'cutoff': 10,
                 'definition': 'PHQ-9/GAD-7 >=10 screening, following Mental-Perceiver'})
                for row in json.loads(path.read_text(encoding='utf-8'))}
    if task == 'dvlog':
        path = data / 'dvlog-dataset/labels.csv'
        with path.open(encoding='utf-8-sig', newline='') as handle:
            return {row['index']: (int(row['label'].strip().lower() == 'depression'),
                    {'path': str(path), 'column': 'label', 'native_value': row['label']})
                    for row in csv.DictReader(handle)}
    return {}


def mmpsy_partitions(root, items):
    """Freeze a derived public-subset split; never reconstruct a native split."""
    source = root / 'annotations/evidence/official_sources/mmpsy_wav_select.json'
    rows = json.loads(source.read_text(encoding='utf-8'))
    identifiers = [row['user_id'] for row in rows]
    annotation_ids = [item['participant'] for item in items]
    if len(identifiers) != 1275 or len(set(identifiers)) != len(identifiers):
        raise ValueError('Expected the 1,275 unique released MMPsy participants')
    if len(set(annotation_ids)) != len(annotation_ids) or set(annotation_ids) != set(identifiers):
        raise ValueError('MMPsy annotations must cover the same released participant set')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    path = root / 'manifests/sources/mmpsy_joint811_seed27.json'
    protocol = 'mmpsy-public-joint811-seed27-v1'
    if path.is_file():
        frozen = json.loads(path.read_text(encoding='utf-8'))
        if frozen['protocol_id'] != protocol or frozen['source_sha256'] != source_hash:
            raise ValueError('Frozen MMPsy source/protocol changed; do not silently replace the split')
    else:
        from sklearn.model_selection import train_test_split
        rows.sort(key=lambda row: row['user_id'])
        strata = [str(int(float(row['phq-label']['score']) >= 10)) +
                  str(int(float(row['gad-label']['score']) >= 10)) for row in rows]
        indices = list(range(len(rows)))
        train, heldout = train_test_split(indices, train_size=1020, random_state=27, stratify=strata)
        dev, test = train_test_split(heldout, train_size=128, random_state=27,
                                     stratify=[strata[index] for index in heldout])
        assignment = {index: split for split, subset in [('train', train), ('dev', dev), ('test', test)]
                      for index in subset}
        frozen = {'protocol_id': protocol, 'source_sha256': source_hash,
                  'source': 'https://github.com/shuyeit/mmpsy-data',
                  'unit': 'released user_id; all answers stay together', 'seed': 27,
                  'ratio': [8, 1, 1], 'stratification': 'joint PHQ-9>=10 and GAD-7>=10',
                  'algorithm': 'two-stage sklearn train_test_split; IDs sorted before splitting',
                  'sklearn_version': __import__('sklearn').__version__,
                  'scope': 'Derived public-subset partition, not the unavailable original participant lists',
                  'references': ['https://arxiv.org/abs/2408.12088',
                                 'https://doi.org/10.3389/fpsyt.2026.1752423'],
                  'participants': [{'participant': row['user_id'], 'split': assignment[index],
                                    'depression': int(strata[index][0]), 'anxiety': int(strata[index][1])}
                                   for index, row in enumerate(rows)]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding='utf-8')
    members = frozen['participants']
    partitions = {row['participant']: row['split'] for row in members}
    if len(members) != 1275 or set(partitions) != set(identifiers):
        raise ValueError('Frozen MMPsy partition has duplicate, missing or unknown participants')
    if Counter(partitions.values()) != Counter(train=1020, dev=128, test=127):
        raise ValueError('Frozen MMPsy partition counts changed')
    source_labels = {row['user_id']: (int(float(row['phq-label']['score']) >= 10),
                                    int(float(row['gad-label']['score']) >= 10)) for row in rows}
    if any(source_labels[row['participant']] != (row['depression'], row['anxiety']) for row in members):
        raise ValueError('Frozen MMPsy labels differ from the released questionnaire scores')
    for split in ('train', 'dev', 'test'):
        if {str(row['depression']) + str(row['anxiety']) for row in members if row['split'] == split} != {'00', '01', '10', '11'}:
            raise ValueError('Every MMPsy split must include all four joint strata')
    provenance = {'protocol_id': protocol, 'manifest': str(path.relative_to(root)),
                  'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'seed': 27, 'unit': 'participant', 'ratio': [8, 1, 1],
                  'stratification': frozen['stratification'], 'scope': frozen['scope']}
    return partitions, provenance


def text_views(item, task):
    views = []
    for turn in item.get('turns', []):
        text = turn.get('text', '').strip()
        if not text:
            continue
        discrete = task.startswith('mmpsy_')
        if discrete:
            start, stop = float(turn['index']), float(turn['index'] + 1)
        else:
            start, stop = turn.get('start'), turn.get('stop')
            if start is None or stop is None or stop <= start:
                continue
            # DAIC's established source-interface window rule is retained.
            if task == 'daic' and stop - start > 20:
                continue
        base = {'event': turn['event'], 'start': start, 'stop': stop,
                'speaker': turn.get('speaker', 'unknown'), 'version': item.get('source_version', 'released'),
                'source': turn.get('source', item['source']), 'available': True,
                'coordinate_unit': 'turn_index' if discrete else 'second'}
        if task == 'ptsd':
            base['seed_eligible_unknown_speaker'] = True
        views.append(base | {'id': turn['view_id'], 'modality': 'T', 'text': text})
        if task in ('daic', 'ptsd', 'eatd'):
            path = turn.get('audio_path', item.get('audio_path'))
            if path:
                views.append(base | {'id': turn['view_id'].rsplit(':', 1)[0] + ':A',
                                      'modality': 'A', 'source': path, 'path': path,
                                      'available': Path(path).is_file()})
    return views


def feature_views(item):
    views = []
    for asset in item.get('features', []):
        path = Path(asset['path'])
        if path.suffix != '.npy' or not path.is_file():
            continue
        name, n_rows = asset['modality'], int(asset['shape'][0])
        timebase = asset.get('timebase') or {}
        step = float(timebase.get('seconds_per_row', 1))
        for first in range(0, n_rows, 20):
            last = min(n_rows, first + 20)
            views.append({'id': f'{name}_{first}_{last}:F', 'event': f'{name}_{first}_{last}',
                          'start': first * step, 'stop': last * step, 'speaker': 'unknown',
                          'modality': 'F', 'source': str(path), 'version': 'released',
                          'available': True, 'features': {name: {'path': str(path), 'first': first, 'last': last}},
                          'coordinate_unit': 'second' if timebase else 'feature_row'})
    return views


def reader_rows(item, rec, counters):
    common = {'family': rec['family'], 'record_id': rec['id'], 'participant': rec['participant'],
              'split': rec['split'], 'annotation_complete': False}
    view_lookup = {view['id']: view for view in rec['views']}
    result = []
    if item.get('turns'):
        turns = item['turns']
        for index, turn in enumerate(turns):
            view_id, targets = turn['view_id'], turn.get('records', [])
            if view_id not in view_lookup or not targets:
                continue
            text = view_lookup[view_id]['text']
            valid = []
            for target in targets:
                start, stop = target['start'], target['stop']
                if target.get('unit') != 'character' or not 0 <= start < stop <= len(text):
                    counters['invalid_text_boundaries'] += 1
                    continue
                if target.get('quote') and text[start:stop] != target['quote']:
                    counters['nonmatching_quote'] += 1
                    continue
                valid.append({'start': start, 'stop': stop, 'unit': 'character', 'fields': target['fields']})
            if not valid:
                continue
            if len(valid) > 8:
                counters['windows_over_eight_records'] += 1
                continue
            # Context is visible source text only, without additional target boundaries.
            context = [other['view_id'] for other in turns[max(0, index - 2):index + 2]
                       if other['view_id'] in view_lookup]
            result.append(common | {'view_id': view_id, 'context_view_ids': context, 'records': valid})
    else:
        for view in rec['views']:
            name, source = next(iter(view['features'].items()))
            first, last = source['first'], source['last']
            valid = []
            for target in item.get('records', []):
                if target.get('modality') != name or target.get('unit') != 'feature_row':
                    continue
                # Source ranges must be contiguous and wholly inside the exposed group.
                ranges = target.get('source_row_ranges', [[target['start'], target['stop']]])
                if ranges != [[target['start'], target['stop']]]:
                    continue
                if first <= target['start'] < target['stop'] <= last:
                    valid.append({'start': target['start'] - first, 'stop': target['stop'] - first,
                                  'unit': 'position', 'fields': {key: target['fields'][key]
                                      for key in ('polarity', 'status') if key in target['fields']}})
            if valid and len(valid) <= 8:
                result.append(common | {'view_id': view['id'], 'context_view_ids': [view['id']], 'records': valid})
    return result


def prepare(root, data):
    annotation_root = root / 'annotations/evidence'
    cross = json.loads((annotation_root / 'cross_release_participants.json').read_text())
    conflicting = {row['participant'] for row in cross if row['different_partition']}
    report, all_reader = {}, []
    for task, name in DATASETS.items():
        path = annotation_root / name / (name + '_all.jsonl')
        source = jsonl(path)
        meta, items = source[0], source[1:]
        counters = Counter()
        blockers = []
        if task == 'modma':
            blockers.append(meta.get('blocked_reason', 'MODMA source recordings unavailable'))
        if blockers:
            report[task] = {'status': 'blocked', 'blockers': blockers, 'source_samples': len(items)}
            continue
        try:
            labels = label_map(root, data, task)
        except FileNotFoundError as error:
            report[task] = {'status': 'blocked', 'blockers': [str(error)]}
            continue
        is_mmpsy = task.startswith('mmpsy_')
        partitions, partition_provenance = mmpsy_partitions(root, items) if is_mmpsy else ({}, meta.get('partition_policy'))
        records, reader = [], []
        for item in items:
            sid = str(item.get('participant') or item.get('source_sample') or item['record_id'])
            label_key = sid if task != 'dvlog' else str(item['record_id'])
            if label_key not in labels:
                counters['missing_native_label'] += 1
                continue
            if task == 'ptsd' and sid in conflicting:
                counters['cross_release_partition_conflict_excluded'] += 1
                continue
            label, provenance = labels[label_key]
            if label not in (0, 1):
                raise ValueError(f'Invalid native binary label: {task}/{sid}')
            split = partitions.get(sid) if is_mmpsy else item['split']
            if split not in ('train', 'dev', 'test', 'unused'):
                counters['missing_partition'] += 1
                continue
            views = feature_views(item) if task == 'dvlog' else text_views(item, task)
            family = 'daic' if task in ('daic', 'ptsd') else ('mmpsy' if is_mmpsy else task)
            record_id = 'e_daic_' + sid if task == 'ptsd' else str(item['record_id'])
            rec = {'id': record_id, 'participant': sid, 'family': family, 'split': split,
                   'label': label, 'label_provenance': provenance, 'views': views,
                   'smoke_only': False, 'partition_provenance': partition_provenance,
                   'identity_provenance': 'released source sample; verified identity clusters unavailable'
                       if task == 'dvlog' else 'released participant identifier'}
            records.append(rec)
            reader.extend(reader_rows(item, rec, counters))
            if task == 'dvlog':
                counters['noncontiguous_feature_spans'] += sum(
                    target.get('source_row_ranges', [[target['start'], target['stop']]]) != [[target['start'], target['stop']]]
                    for target in item.get('records', []))
        dump(root / f'manifests/tasks/{task}.jsonl', records)
        # Task-specific copies avoid confusing native DAIC/E-DAIC source windows.
        dump(root / f'manifests/reader/{task}.jsonl', reader)
        if task != 'mmpsy_anxiety':  # Both tasks share the same participant and source annotations.
            all_reader.extend(reader)
        cohort = {}
        supervised_ids = {row['record_id'] for row in reader}
        for split in ('train', 'dev', 'test', 'unused'):
            candidates = [rec for rec in records if rec['split'] == split]
            candidates.sort(key=lambda rec: rec['id'] not in supervised_ids)
            cohort[split] = [next((r['id'] for r in candidates if r['label'] == label and len(r['views']) >= 3), None)
                             for label in (0, 1)]
            cohort[split] = [value for value in cohort[split] if value is not None]
        report[task] = {'status': 'ready',
                        'records': len(records), 'splits': dict(Counter(r['split'] for r in records)),
                        'reader_windows': len(reader), 'reader_splits': dict(Counter(r['split'] for r in reader)),
                        'smoke_only': False, 'smoke_cohort_ids': cohort, 'excluded': dict(counters),
                        'partition_provenance': partition_provenance,
                        'source': str(path)}
        if task == 'ptsd' and counters['missing_native_label']:
            report[task]['formal_evaluation_blockers'] = [
                'Released metadata has no native PTSD labels for 56 official test participants; training/dev smoke is available.']
        if is_mmpsy:
            report[task]['evaluation_scope'] = 'Derived public-subset protocol; do not compare as the original full MMPsy split.'
            report[task]['label_counts'] = {split: dict(Counter(str(r['label']) for r in records if r['split'] == split))
                                          for split in ('train', 'dev', 'test')}
    dump(root / 'manifests/reader/items.jsonl', all_reader)
    report_path = root / 'manifests/table4_preparation.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path, help='Dataset root; defaults to the DAIC location in configs/experiments.json')
    args = parser.parse_args()
    if args.data_root is None:
        config = json.loads((args.root / 'configs/experiments.json').read_text(encoding='utf-8'))
        args.data_root = Path(config['tasks']['daic']['splits']['train']['label_csv']).parent.parent
    prepare(args.root, args.data_root)
