"""Source-supervised text Reader for the EMER qualitative transfer diagnostic.

This reuses the project's actual supervised Reader. EMER answers and dataset
explanations never enter Reader fitting or its observed transcript inputs.
"""
import copy
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

import torch

from ..reader import ReaderRuntime
from .data import digest, load_records, read_jsonl, write_json
from .reader_training import fit_reader, index_reader_positions


FrozenReader = ReaderRuntime


def _settings(root):
    text = (root / 'scripts/ega/ega_eatd_dep_qo.sh').read_text(encoding='utf-8-sig')
    marker = "<<'SETTINGS'"
    if marker not in text:
        raise ValueError('Cannot locate the declared EATD Reader settings')
    return json.loads(text.split(marker, 1)[1].split('\nSETTINGS', 1)[0])


def _hash(seed, value):
    return hashlib.sha256(f'emer-reader-source-v1:{seed}:{value}'.encode()).hexdigest()


def _sample(annotations, cap, seed):
    """Alternate source families and participant queues, without reading labels."""
    grouped = defaultdict(lambda: defaultdict(list))
    for annotation in annotations:
        grouped[annotation['family']][annotation['participant']].append(annotation)
    queues = {}
    for family, participants in grouped.items():
        people = []
        for participant in sorted(participants, key=lambda p: _hash(seed, (family, p))):
            rows = sorted(participants[participant], key=lambda a: _hash(seed,
                (a['family'], a['record_id'], a['view_id'])))
            people.append(deque(rows))
        queues[family] = deque(people)
    chosen = []
    while len(chosen) < cap and any(queues.values()):
        for family in sorted(queues):
            if len(chosen) >= cap or not queues[family]:
                continue
            person = queues[family].popleft()
            chosen.append(person.popleft())
            if person:
                queues[family].append(person)
    return chosen


def source_reader(backbone, outdir, seed=27, *, root=None,
                  max_train_windows=96, max_dev_windows=24, epochs=100, patience=10):
    """Fit/freeze the existing Reader on source TRAIN/DEV annotations only.

    Returns a ReaderRuntime (also exported as FrozenReader). The parent adapter
    must expose the tokenizer, model.get_input_embeddings(), and stable spec.
    Reusing outdir resumes the exact same training checkpoint; changing the
    sampled windows or settings in that directory is rejected.
    """
    for name, value in (('seed', seed), ('max_train_windows', max_train_windows),
                        ('max_dev_windows', max_dev_windows), ('epochs', epochs),
                        ('patience', patience)):
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'seed' else 1):
            raise ValueError(f'{name} must be a positive integer (seed may be zero)')
    root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    settings = _settings(root)
    config_path = Path(settings['config'])
    config = json.loads((config_path if config_path.is_absolute() else root/config_path).read_text(encoding='utf-8-sig'))
    reader_config = copy.deepcopy(settings['reader'])
    annotation_path = Path(reader_config['annotations'])
    annotation_path = annotation_path if annotation_path.is_absolute() else root/annotation_path
    annotations = read_jsonl(annotation_path)
    records = []
    for task in settings['reader_sources']:
        if task not in ('daic', 'mmpsy_depression'):
            raise ValueError('EMER qualitative Reader permits only the declared DAIC/MMPsy sources')
        records.extend(r for r in load_records(root, config['tasks'][task])
                       if r['split'] in ('train', 'dev'))
    lookup = {(r['family'], r['id']): r for r in records}
    eligible = {'train': [], 'dev': []}
    for annotation in annotations:
        key = annotation['family'], annotation['record_id']
        rec = lookup.get(key)
        if rec is None:
            continue
        split = rec['split']
        if annotation['split'] != split or annotation['participant'] != rec['participant']:
            raise ValueError('Source Reader annotation partition/participant mismatch')
        views = {v['id']: v for v in rec['views']}
        focal = views.get(annotation['view_id'])
        if focal is None or focal['modality'] != 'T':
            continue
        context_ids = annotation.get('context_view_ids', [annotation['view_id']])
        if any(i not in views or views[i]['modality'] != 'T' for i in context_ids):
            continue
        eligible[split].append(annotation)
    sampled = {'train': _sample(eligible['train'], max_train_windows, seed),
               'dev': _sample(eligible['dev'], max_dev_windows, seed)}
    if not all(sampled.values()):
        raise ValueError('Source text annotations are required in both TRAIN and DEV')
    selected = sampled['train']+sampled['dev']
    selected_keys = {(a['family'], a['record_id']) for a in selected}
    source_records = []
    for record in records:
        key = record['family'], record['id']
        if key not in selected_keys:
            continue
        item = copy.deepcopy(record)
        relevant = [a for a in selected if (a['family'], a['record_id']) == key]
        ids = {i for a in relevant for i in a.get('context_view_ids', [a['view_id']])}
        item['views'] = [v for v in item['views'] if v['modality'] == 'T' and v['id'] in ids]
        source_records.append(item)
    source_annotations = outdir/'source_annotations.jsonl'
    reader_config.update(annotations=str(source_annotations), epochs=epochs, patience=patience,
                         memory_modalities='supervised', max_windows=None)
    provenance = {
        'schema': 'emer-qualitative-source-reader-v1', 'seed': seed,
        'scope': 'Small source-supervised text Reader for qualitative transfer; not full main-experiment weights.',
        'sources': list(settings['reader_sources']), 'partitions': ['train', 'dev'],
        'source_annotation_path': str(annotation_path), 'source_annotation_sha256': hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        'source_settings': 'scripts/ega/ega_eatd_dep_qo.sh',
        'sampling': 'Fixed hash of participant and source-window identity; alternate source families and participants; no task labels used.',
        'requested_caps': {'train': max_train_windows, 'dev': max_dev_windows},
        'excluded': 'No EMER answer, clean_list, solution/context/think, EMER gold evidence, test partition or generated emotion relation targets.',
        'reader_config': reader_config,
        'windows': {split: [{k: a[k] for k in ('family', 'record_id', 'participant', 'split', 'view_id')}
                            for a in rows] for split, rows in sampled.items()},
        'selected_annotations_sha256': digest(selected),
    }
    provenance_path = outdir/'source_reader_provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text(encoding='utf-8')) != provenance:
        raise ValueError('Source Reader inputs/settings changed; use another explicit output directory')
    source_annotations.write_text(''.join(json.dumps(a, ensure_ascii=False)+'\n' for a in selected), encoding='utf-8')
    write_json(provenance_path, provenance)
    index_reader_positions(source_records, reader_config)
    train = [r for r in source_records if r['split'] == 'train']
    dev = [r for r in source_records if r['split'] == 'dev']
    checkpoint_path = outdir/'reader_training.pt'
    bundle = fit_reader(backbone, train, dev, reader_config, seed,
                        checkpoint_path=checkpoint_path, resume=checkpoint_path.exists())
    torch.save(bundle, outdir/'reader_bundle.pt')
    runtime = ReaderRuntime(backbone, bundle, include_relation_probabilities=True)
    runtime.provenance = provenance
    backbone.reader = runtime
    write_json(outdir/'reader_summary.json', {
        'schema': 'emer-qualitative-source-reader-result-v1', 'fingerprint': runtime.fingerprint,
        'training_windows': bundle['training_windows'], 'development_windows': bundle['development_windows'],
        'selected_epoch': bundle['epochs'], 'presence_threshold': bundle['threshold'],
        'record_confidence_threshold': bundle['record_confidence_threshold'],
        'source_dev_span_relation_error': bundle['selection_error'],
        'supervised_modalities': bundle['supervised_modalities'],
        'provenance': str(provenance_path), 'history': bundle['history'],
        'limits': 'Out-of-domain text transfer only; no visual Reader training or EMER evidence-span accuracy claim.',
    })
    return runtime


def transcript_views(item):
    """Create source-linked T views from actual ASR/manual transcript fields.

    Accepts existing T views, transcript_segments, segments, or transcript text.
    A raw transcript fallback requires the real clip duration. Dataset reference
    explanations are deliberately not recognized as a transcript field.
    """
    if item.get('views'):
        views = [copy.deepcopy(v) for v in item['views'] if v.get('modality') == 'T']
        if views:
            return views
    payload = item.get('transcript_segments', item.get('segments'))
    transcript = item.get('transcript')
    if payload is None and isinstance(transcript, dict):
        payload = transcript.get('segments')
        transcript = transcript.get('text')
    if payload is None and isinstance(transcript, list):
        payload = transcript
    if payload is None:
        if not isinstance(transcript, str) or not transcript.strip():
            return []
        stop = item.get('duration', item.get('duration_seconds'))
        if stop is None:
            raise ValueError('An actual transcript without segments requires the real clip duration')
        payload = [{'start': 0., 'end': stop, 'text': transcript}]
    clip_id = str(item.get('id', item.get('video', 'emer_clip')))
    source = item.get('transcript_source', item.get('video_path', item.get('server_video_path', item.get('video'))))
    if not source:
        raise ValueError('EMER transcript requires its original source path')
    views = []
    for i, segment in enumerate(payload):
        if not isinstance(segment, dict) or not isinstance(segment.get('text'), str):
            raise ValueError('Transcript segments require observed text and original timestamps')
        text = segment['text'].strip()
        if not text:
            continue
        start, stop = float(segment['start']), float(segment.get('stop', segment.get('end')))
        if not 0 <= start < stop:
            raise ValueError('Invalid observed transcript interval')
        views.append({'id': f'{clip_id}:transcript_{i}:T', 'event': f'transcript_{i}',
                      'modality': 'T', 'text': text, 'start': start, 'stop': stop,
                      'speaker': segment.get('speaker', 'unknown'), 'source': str(source),
                      'version': item.get('transcript_version', 'actual_media_transcription'),
                      'available': True, 'coordinate_unit': 'seconds'})
    return views


def read_transcript(reader, item):
    """Return actual frozen Reader records for the observed transcript only."""
    views = transcript_views(item)
    if not views:
        return []
    index_reader_positions([{'views': views}], reader.config)
    return reader.read(views)
