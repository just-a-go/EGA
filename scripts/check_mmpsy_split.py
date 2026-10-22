"""Validate frozen MMPsy partitions and reader isolation without loading a model."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    frozen_path = ROOT / 'manifests/sources/mmpsy_joint811_seed27.json'
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    source_path = ROOT / 'annotations/evidence/official_sources/mmpsy_wav_select.json'
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == frozen['source_sha256']
    raw = json.loads(source_path.read_text(encoding='utf-8'))
    labels = {row['user_id']: {'depression': int(float(row['phq-label']['score']) >= 10),
                              'anxiety': int(float(row['gad-label']['score']) >= 10)} for row in raw}
    assignment = {row['participant']: row['split'] for row in frozen['participants']}
    assert len(assignment) == len(frozen['participants']) == len(raw) == len(labels) == 1275
    assert set(assignment) == set(labels)
    groups = {split: {sid for sid, value in assignment.items() if value == split}
              for split in ('train', 'dev', 'test')}
    assert {split: len(group) for split, group in groups.items()} == {'train': 1020, 'dev': 128, 'test': 127}
    assert not (groups['train'] & groups['dev'] or groups['train'] & groups['test'] or groups['dev'] & groups['test'])
    joint_counts = {split: dict(Counter(str(labels[sid]['depression']) + str(labels[sid]['anxiety'])
                                       for sid in group)) for split, group in groups.items()}
    assert all(set(counts) == {'00', '01', '10', '11'} for counts in joint_counts.values())
    report = {'protocol_id': frozen['protocol_id'], 'participants': 1275,
              'split_counts': {split: len(group) for split, group in groups.items()},
              'joint_label_counts': joint_counts, 'tasks': {}, 'participant_overlap': 0,
              'model_loaded': False, 'gpu_training_started': False}
    for dimension in ('depression', 'anxiety'):
        task = 'mmpsy_' + dimension
        records = jsonl(ROOT / f'manifests/tasks/{task}.jsonl')
        assert len(records) == len({row['id'] for row in records}) == 1275
        assert {row['participant'] for row in records} == set(assignment)
        for row in records:
            assert row['split'] == assignment[row['participant']]
            assert row['label'] == labels[row['participant']][dimension]
            assert row['partition_provenance']['protocol_id'] == frozen['protocol_id']
        reader = jsonl(ROOT / f'manifests/reader/{task}.jsonl')
        by_id = {row['id']: row for row in records}
        for row in reader:
            record = by_id[row['record_id']]
            assert row['participant'] == record['participant']
            assert row['split'] == record['split']
        report['tasks'][task] = {'records': len(records), 'reader_windows': len(reader),
                                'reader_splits': dict(Counter(row['split'] for row in reader))}
        for model in ('QT', 'G3'):
            launcher = (ROOT / f'scripts/ega/ega_mmpsy_{dimension[:3]}_{model.lower()}.sh').read_text()
            settings = json.loads(launcher.split("<<'SETTINGS'", 1)[1].rsplit('SETTINGS', 1)[0])
            assert settings['task'] == task and settings['model'] == model
            assert settings['dataset_partition']['protocol_id'] == frozen['protocol_id']
            assert settings['dataset_partition']['seed'] == frozen['seed'] == 27
    combined = [row for row in jsonl(ROOT / 'manifests/reader/items.jsonl') if row['family'] == 'mmpsy']
    depression_reader = jsonl(ROOT / 'manifests/reader/mmpsy_depression.jsonl')
    anxiety_reader = jsonl(ROOT / 'manifests/reader/mmpsy_anxiety.jsonl')
    assert combined == depression_reader == anxiety_reader
    report['status'] = 'passed'
    (ROOT / 'manifests/mmpsy_split_validation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
