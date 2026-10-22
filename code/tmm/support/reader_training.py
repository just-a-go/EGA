"""Participant-balanced supervised fitting, development selection, then freezing."""
import copy
import json
import random

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .data import digest, ObservationError
from . import grpo
from ..reader import (EvidenceReader, ReaderRuntime, consistency_loss, decode, human_loss,
                     reader_fingerprint, _predicted_span)
from .reader_inputs import (SourceEncoder, annotations_for, load_annotations, validate_config,
                            reader_training_settings)


def index_reader_positions(records, config):
    """Metadata reservations are computed at indexing time, never in action scoring."""
    import math
    import time
    for rec in records:
        started = time.perf_counter()
        for view in rec['views']:
            if view['modality'] == 'T':
                positions = len(view['text'].encode('utf-8'))  # Conservative byte-token upper bound.
            elif view['modality'] == 'A':
                positions = math.ceil((view['stop']-view['start'])/config['audio_frame_seconds'])+1
            else:
                positions = sum(s['last']-s['first'] for s in view['features'].values())
            view['reader_positions'] = positions+1  # One source metadata row.
        rec['index_seconds'] = rec.get('index_seconds', 0.)+time.perf_counter()-started


def prepare_examples(backbone, rows, config, excluded=None, allowed_modalities=None):
    encoder = SourceEncoder(backbone, config)
    examples = []
    def exclude(row, reason, error):
        entry = {key: row['annotation'].get(key) for key in ('family', 'record_id', 'view_id')}
        entry.update(reason=reason, error=str(error)[:300])
        if excluded is not None:
            excluded.append(entry)
        print('reader_excluded: '+json.dumps(entry, ensure_ascii=False), flush=True)
    for row in rows:
        row = copy.deepcopy(row)
        if allowed_modalities is not None:
            row['views'] = [v for v in row['views'] if v['modality'] in allowed_modalities]
            if not any(v['id'] == row['annotation']['view_id'] for v in row['views']):
                exclude(row, 'source_modality_unsupervised',
                        'Reader focal modality has no source-training supervision')
                continue
        try:
            sources = [encoder.encode(view) for view in row['views']]
        except (ObservationError, FileNotFoundError) as exc:
            exclude(row, 'source_encoding_failure', exc)
            continue
        # The largest equivalent presentation includes a repeat and marker.
        if sum(len(s['values'])+1 for s in sources)+len(sources[0]['values'])+1 > config['max_positions']:
            exclude(row, 'context_positions_exceeded',
                    'Annotated reader context exceeds max_positions, including presentation variants; no truncation')
            continue
        focal = next(s for s in sources if s['view']['id'] == row['annotation']['view_id'])
        for target in row['annotation']['records']:
            if target['unit'] == 'character':
                if focal['view']['modality'] != 'T' or focal['offsets'] is None:
                    raise ValueError('Character span annotations require a fast text tokenizer with source offsets')
                if target['stop'] > len(focal['view']['text']):
                    raise ValueError('Annotated character span exceeds source text')
                covered = [i for i, (a,b) in enumerate(focal['offsets']) if b > target['start'] and a < target['stop']]
                if not covered:
                    raise ValueError('Annotated character span maps to no source tokens')
                target['start'], target['stop'] = covered[0], covered[-1]+1
                target['unit'] = 'position'
            elif focal['view']['modality'] == 'T':
                tokenizer_id = getattr(backbone.tokenizer, 'name_or_path', None)
                if not tokenizer_id or row['annotation'].get('tokenizer') != tokenizer_id:
                    raise ValueError('Text position annotations must name the exact backbone tokenizer; prefer character spans')
            if target['stop'] > len(focal['values']):
                raise ValueError('Annotated reader boundary exceeds source positions')
            if focal['view']['modality'] == 'F':
                first, last = target['start'], target['stop']-1
                if focal['coordinates'][first][0] != focal['coordinates'][last][0]:
                    raise ValueError('Feature evidence cannot cross modality coordinate systems')
        examples.append(row | {'sources': sources, 'focal': focal})
    return examples


def participant_terms(model, examples, config, variants=True):
    # Compute participant-wide denominators without retaining every window's graph.
    counts = {}
    render_count = 3 if variants else 1
    for example in examples:
        annotation = example['annotation']
        targets = annotation['records']
        counts['presence'] = counts.get('presence', 0)+(8 if annotation.get('annotation_complete', False) else len(targets))*render_count
        for target in targets:
            for key in ('start','stop', *target['fields']):
                counts[key] = counts.get(key, 0)+render_count
    for example in examples:
        sums, consistency = {}, []
        annotation = example['annotation']
        left = model(example['sources'], annotation['view_id'])
        class_weights = getattr(model, 'supervision_weights', None)
        boundary = config.get('boundary_likelihood', 'legacy_independent')
        parts, n, left_match = human_loss(left, annotation, config['ontology'], class_weights, boundary,
                                          source=example['focal'])
        for key in parts:
            sums[key] = sums.get(key, 0)+parts[key]
        if variants:
            for renderer in (1, 2):
                right = model(example['sources'], annotation['view_id'], renderer)
                parts, n, right_match = human_loss(right, annotation, config['ontology'], class_weights, boundary,
                                               source=example['focal'])
                for key in parts:
                    sums[key] = sums.get(key, 0)+parts[key]
                consistency.append(consistency_loss(left, right, left_match, right_match,
                                                     annotation.get('annotation_complete', False), boundary, example['focal']))
        weights = config.get('loss_weights', {})
        value = sum(weights.get(key, 1.)*sums[key]/counts[key] for key in sums)
        if consistency:
            value = value+config['consistency_weight']*torch.stack(consistency).mean()/len(examples)
        yield value


def participant_loss(model, examples, config, variants=True):
    return sum(participant_terms(model, examples, config, variants))


def group_examples(examples):
    groups = {}
    for example in examples:
        groups.setdefault(example['participant'], []).append(example)
    return groups


def reader_participant_batches(count, settings, seed=27, epoch=0):
    """Local reproducible order; a partial final batch keeps the same participant coefficient."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('Reader batching requires positive participant count')
    mode = settings['mode']
    resolved = reader_training_settings({'training_mode': mode,
        **({'participant_batch_size': settings['participant_batch_size']}
           if mode != 'fullbatch' else {})})
    order = list(range(count))
    if mode == 'fullbatch':
        return [order], count
    random.Random(f'reader-participant-minibatch-v1:{int(seed)}:{int(epoch)}').shuffle(order)
    size = resolved['participant_batch_size']
    return [order[first:first+size] for first in range(0, count, size)], size


def relation_weights(examples, ontology, mode):
    """Balance observed relation labels without inventing negative evidence."""
    counts = {name: [0]*len(categories) for name, categories in ontology.items()}
    for example in examples:
        for target in example['annotation']['records']:
            for name, category in target['fields'].items():
                counts[name][ontology[name].index(category)] += 1
    if mode not in ('none', 'sqrt_inverse_frequency'):
        raise ValueError('Unsupported reader relation_class_balance')
    weights = {name: [1.]*len(categories) for name, categories in ontology.items()}
    if mode == 'sqrt_inverse_frequency':
        for name, values in counts.items():
            values = np.asarray(values, dtype=float)
            total = values.sum()
            if total:
                weight = np.sqrt(total/np.maximum(1., values))
                weight /= float(np.dot(values, weight)/total)
                weights[name] = weight.tolist()
    return weights, counts


def span_relation_error(predicted, desired, annotation_complete, record_threshold=0., abstention_cost=1.):
    """Score annotated spans/fields; unannotated partial windows are not negatives."""
    cost = np.ones((len(desired), len(predicted)), dtype=float)
    overlaps = np.zeros_like(cost, dtype=bool)
    accepted = np.array([r.get('confidence', 1.) >= record_threshold for r in predicted], dtype=bool)
    for i, target in enumerate(desired):
        for j, record in enumerate(predicted):
            first, stop = _predicted_span(record)
            overlap = max(0, min(target['stop'], stop)-max(target['start'], first))
            union = target['stop']-target['start']+stop-first-overlap
            overlaps[i, j] = overlap > 0
            if not overlap:
                continue  # A known annotated record was missed, not a negative label for this prediction.
            span_error = 1.-overlap/union
            field_errors = [record['fields'].get(key, 'unknown') != value
                            for key, value in target['fields'].items()]
            cost[i, j] = ((span_error+float(np.mean(field_errors)))/2 if field_errors else span_error) if accepted[j] else abstention_cost
    a, b = linear_sum_assignment(cost)
    missing = len(desired)-len(a)
    accepted_matches = sum(accepted[j] and overlaps[i, j] for i, j in zip(a, b))
    extras = int(accepted.sum())-accepted_matches if annotation_complete else 0
    return float((cost[a, b].sum()+missing+extras)/max(1, len(desired)))


def select_threshold(model, examples, config, supported):
    # Ignore unmatched predictions in incompletely annotated windows.
    cached = []
    with torch.no_grad():
        for example in examples:
            output = model(example['sources'], example['annotation']['view_id'])
            cached.append((example, output))
    trials = []
    for threshold in config['presence_thresholds']:
        decoded = []
        for example, output in cached:
            source, annotation = example['focal'], example['annotation']
            predicted = decode(output, source, config['ontology'], threshold,
                               supported.get(source['view']['modality'], []),
                               config.get('relation_confidence_threshold', .7),
                               boundary_likelihood=config.get('boundary_likelihood', 'legacy_independent'),
                               speaker_normalization=config.get('speaker_normalization', 'literal'))
            decoded.append((example, predicted))
        confidence_values = {r['confidence'] for example, records in decoded for r in records}
        record_thresholds = sorted({0., *confidence_values, max(confidence_values, default=0.)+1e-6})
        for record_threshold in record_thresholds:
            errors, quality = {}, {}
            for example, predicted in decoded:
                annotation, desired = example['annotation'], example['annotation']['records']
                accepted = [r for r in predicted if r['confidence'] >= record_threshold]
                used, matches = set(), 0
                for target in desired:
                    for i, record in enumerate(accepted):
                        if i not in used and _predicted_span(record) == (target['start'], target['stop']):
                            used.add(i)
                            matches += 1
                            break
                misses = len(desired)-matches
                extras = len(accepted)-matches if annotation.get('annotation_complete', False) else 0
                errors.setdefault(example['participant'], []).append((misses+extras)/max(1,len(desired)))
                quality.setdefault(example['participant'], []).append(span_relation_error(
                    predicted, desired, annotation.get('annotation_complete', False), record_threshold,
                    config.get('record_abstention_cost', .5)))
            score = float(np.mean([np.mean(v) for v in errors.values()]))
            quality_score = float(np.mean([np.mean(v) for v in quality.values()]))
            trials.append({'threshold': threshold, 'record_confidence_threshold': record_threshold,
                           'dev_record_error': score, 'span_relation_error': quality_score})
    best = min(trials, key=lambda x: (x['span_relation_error'], -x['record_confidence_threshold'],
                                    x['dev_record_error'], -x['threshold']))
    return best['threshold'], trials


def fit_reader(backbone, train, dev, config, seed=27, selected=None,
               checkpoint_path=None, resume=False):
    validate_config(config)
    training_settings = reader_training_settings(config)
    relation_conditioning = config.get('relation_conditioning', 'query_only')
    if relation_conditioning not in ('query_only', 'predicted_span_mean'):
        raise ValueError('Unsupported Reader relation conditioning')
    train_keys = {(r['family'], r['participant']) for r in train}
    dev_keys = {(r['family'], r['participant']) for r in dev}
    if train_keys & dev_keys:
        raise ValueError('Reader training and development participants overlap')
    if any(r.get('split') in ('test', 'unused') for r in train+dev):
        raise ValueError('Test/audit participants cannot fit or select the evidence reader')
    annotations = load_annotations(config['annotations'])
    train_rows = annotations_for(train, annotations, config)
    dev_rows = annotations_for(dev, annotations, config)
    supervised_modalities = sorted({next(v['modality'] for v in row['views']
                                         if v['id'] == row['annotation']['view_id']) for row in train_rows})
    allowed_modalities = (supervised_modalities if config.get('memory_modalities', 'all_observed') == 'supervised'
                          else None)
    if config.get('max_windows'):
        def bounded(rows, records):
            groups = [[x for x in rows if x['annotation']['record_id'] == rec['id']
                       and x['annotation']['family'] == rec['family']] for rec in records]
            chosen = []
            while any(groups) and len(chosen) < config['max_windows']:
                for group in groups:
                    if group and len(chosen) < config['max_windows']:
                        chosen.append(group.pop(0))
            return chosen
        train_rows = bounded(train_rows, train)
        dev_rows = bounded(dev_rows, dev)
    expected = grpo.signature(('reader-boundary-memory-v2', config, seed, selected,
                              getattr(backbone, 'spec', {}), train_rows, dev_rows, supervised_modalities))
    restored = grpo.load_checkpoint(checkpoint_path, resume, 'ega-reader-training-v1', expected)
    if restored and restored.get('bundle'):
        backbone.reader = ReaderRuntime(backbone, restored['bundle'])
        grpo.restore_rng(restored['rng'])
        print('reader_resume: completed reader bundle restored', flush=True)
        return restored['bundle']
    excluded = {'train': [], 'dev': []}
    training = prepare_examples(backbone, train_rows, config, excluded['train'], allowed_modalities)
    development = prepare_examples(backbone, dev_rows, config, excluded['dev'], allowed_modalities)
    if not training or (not development and selected is None):
        raise ValueError('EGA requires usable human reader windows for train/dev; '
                         f"excluded train={len(excluded['train'])}, dev={len(excluded['dev'])}")
    if config.get('selection_metric', 'span_relation_error') != 'span_relation_error':
        raise ValueError('Reader checkpoint selection requires span_relation_error')
    relation_threshold = config.get('relation_confidence_threshold', .7)
    if not 0 < relation_threshold < 1:
        raise ValueError('Reader relation_confidence_threshold must lie in (0, 1)')
    if not 0 < config.get('record_abstention_cost', .5) < 1:
        raise ValueError('Reader record_abstention_cost must lie in (0, 1)')
    class_weights, class_counts = relation_weights(
        training, config['ontology'], config.get('relation_class_balance', 'sqrt_inverse_frequency'))
    supported = {}
    for example in training:
        modality = example['focal']['view']['modality']
        supported.setdefault(modality, set()).update(key for r in example['annotation']['records'] for key in r['fields'])
    supported = {k: sorted(v) for k, v in supported.items()}
    torch.manual_seed(seed)
    model = EvidenceReader(config['ontology'], relation_conditioning=relation_conditioning,
                           boundary_likelihood=config.get('boundary_likelihood', 'legacy_independent'))
    model.supervision_weights = class_weights
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    groups, validation = group_examples(training), group_examples(development)
    participant_keys = list(groups)
    participant_groups = list(groups.values())
    best, stale, history, next_epoch, saved, best_epoch, complete = float('inf'), 0, [], 0, None, 0, False
    optimizer_updates = 0
    best_quality, best_threshold, best_record_threshold, best_trials = float('inf'), None, 0., []
    epochs = int(selected['epochs'] if selected else config['epochs'])
    if epochs < 1:
        raise ValueError('Reader epochs must be positive')
    if restored:
        model.load_state_dict(restored['model'])
        optimizer.load_state_dict(restored['optimizer'])
        best, stale, history = restored['best'], restored['stale'], restored['history']
        next_epoch, saved, best_epoch = restored['next_epoch'], restored['best_reader'], restored['best_epoch']
        best_quality = restored.get('best_quality', float('inf'))
        best_threshold, best_trials = restored.get('best_threshold'), restored.get('best_trials', [])
        best_record_threshold = restored.get('best_record_threshold', 0.)
        complete = restored['complete']
        optimizer_updates = restored.get('optimizer_updates', len(history))
        if (restored.get('training_settings', {'mode': 'fullbatch', 'participant_batch_size': None})
                != training_settings):
            raise ValueError('Restored Reader optimizer grouping changed')
        if training_settings['mode'] != 'fullbatch':
            batches, _ = reader_participant_batches(len(groups), training_settings, seed, next_epoch)
            if (restored.get('participant_roster_signature') != digest(participant_keys)
                    or 'optimizer_updates' not in restored
                    or optimizer_updates != next_epoch*len(batches)):
                raise ValueError('Restored Reader participant roster/epoch-boundary update count changed')
        grpo.restore_rng(restored['rng'])
        print(f'reader_resume: next_epoch={next_epoch+1}', flush=True)

    def checkpoint(next_epoch, done, bundle=None):
        if checkpoint_path is not None:
            grpo.save_checkpoint(checkpoint_path, {
                'schema': 'ega-reader-training-v1', 'signature': expected,
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': grpo.rng_state(),
                'next_epoch': next_epoch, 'best': best, 'stale': stale, 'history': history,
                'best_epoch': best_epoch, 'best_reader': saved, 'complete': done,
                'best_quality': best_quality, 'best_threshold': best_threshold, 'best_trials': best_trials,
                'best_record_threshold': best_record_threshold,
                'optimizer_updates': optimizer_updates, 'training_settings': training_settings,
                'participant_roster_signature': digest(participant_keys),
                'excluded_windows': excluded, 'bundle': bundle, 'boundary': 'completed_epoch'})

    for epoch in range(next_epoch, epochs) if not complete else ():
        model.train()
        batches, divisor = reader_participant_batches(len(groups), training_settings, seed, epoch)
        epoch_updates = 0
        order_signature = digest([participant_keys[index] for batch in batches for index in batch])
        for batch_index, indices in enumerate(batches):
            optimizer.zero_grad()
            for index in indices:
                for value in participant_terms(model, participant_groups[index], config):
                    # participant_terms already normalizes that participant's windows.
                    # The last partial batch also uses the fixed configured divisor.
                    (value/divisor).backward()
            optimizer.step()
            optimizer_updates += 1
            epoch_updates += 1
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise RuntimeError('Non-finite reader optimizer update')
            if training_settings['mode'] != 'fullbatch':
                print('reader_optimizer_update: '+json.dumps({'epoch': epoch+1,
                    'optimizer_update': optimizer_updates, 'batch': batch_index+1,
                    'batches': len(batches), 'participants': len(indices),
                    'participant_divisor': divisor, 'participant_order_signature': order_signature}), flush=True)
        model.eval()
        with torch.no_grad():
            value = float(np.mean([float(participant_loss(model, examples, config))
                                   for examples in (validation or groups).values()]))
        if selected is None:
            candidate_threshold, trials = select_threshold(model, development, config, supported)
            best_trial = min(trials, key=lambda t: (t['span_relation_error'], -t['record_confidence_threshold'],
                                                   t['dev_record_error'], -t['threshold']))
            quality, record_threshold = best_trial['span_relation_error'], best_trial['record_confidence_threshold']
        else:
            candidate_threshold, trials, quality = selected['threshold'], [], None
            record_threshold = selected.get('record_confidence_threshold', 0.)
        history.append({'epoch': epoch+1, 'validation_loss': value, 'span_relation_error': quality,
                        'presence_threshold': candidate_threshold})
        history[-1]['record_confidence_threshold'] = record_threshold
        if training_settings['mode'] != 'fullbatch':
            history[-1].update(training_mode=training_settings['mode'],
                participant_batch_size=training_settings['participant_batch_size'],
                epoch_optimizer_updates=epoch_updates, optimizer_updates=optimizer_updates,
                participant_order='seeded_epoch_shuffle', participant_order_signature=order_signature,
                training_participants=len(groups), last_batch_participants=len(batches[-1]),
                participant_gradient_divisor=divisor)
        print(f'reader epoch={epoch+1} validation_loss={value:.6f} span_relation_error={quality}', flush=True)
        if selected is not None or (quality, value) < (best_quality, best):
            best, stale, best_epoch = value, 0, epoch+1
            best_quality, best_threshold, best_trials = quality, candidate_threshold, trials
            best_record_threshold = record_threshold
            saved = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        next_epoch = epoch+1
        complete = (selected is None and stale >= config['patience']) or next_epoch >= epochs
        checkpoint(next_epoch, complete)
        if complete:
            break
    if saved is None:
        raise ValueError('Reader checkpoint contains no completed validation epoch')
    model.load_state_dict(saved)
    model.eval().requires_grad_(False)
    threshold, trials = (selected['threshold'], []) if selected else (best_threshold, best_trials)
    bundle = {'state_dict': saved, 'config': config, 'seed': seed, 'epochs': best_epoch,
              'backbone_spec': copy.deepcopy(getattr(backbone, 'spec', {})),
              'threshold': threshold, 'threshold_trials': trials, 'supported': supported,
              'supervised_modalities': supervised_modalities,
              'validation_loss': best, 'history': history,
              'selection_metric': 'span_relation_error', 'selection_error': best_quality,
              'record_confidence_threshold': best_record_threshold,
              'record_abstention_cost': config.get('record_abstention_cost', .5),
              'decoder_policy': {'context': 'complete_source_sentence', 'merge_overlap_iou': .8,
                                 'boundary_likelihood': config.get('boundary_likelihood', 'legacy_independent'),
                                 'memory_modalities': config.get('memory_modalities', 'all_observed'),
                                 'quality_coordinates': 'original_predicted_span',
                                 'quality_selection': 'source_dev_partial_span_relation_with_abstention',
                                 'partial_supervision_limit': 'Unannotated spans are not negative labels; confidence calibration has no verified negatives unless annotation_complete windows exist.'},
              'relation_class_counts': class_counts, 'relation_class_weights': class_weights,
              'supervision': {split: {'annotated_participants': len(group_examples(examples)),
                                      'available_participants': len(keys),
                                      'annotation_complete_windows': sum(e['annotation'].get('annotation_complete', False) for e in examples),
                                      'annotation_complete_negative_windows': sum(e['annotation'].get('annotation_complete', False) and
                                                                             not e['annotation']['records'] for e in examples)}
                              for split, examples, keys in [('train', training, train_keys),
                                                            ('dev', development, dev_keys)]},
              'train_participants': sorted(train_keys), 'dev_participants': sorted(dev_keys),
              'annotations_sha256': digest([r['annotation'] for r in training+development]),
              'training_windows': len(training), 'development_windows': len(development),
              'excluded_windows': excluded,
              'exclusion_counts': {split: len(items) for split, items in excluded.items()},
              'representation': 'frozen_backbone_text_embeddings; audio_100ms_statistics; temporal_feature_rows'}
    if relation_conditioning != 'query_only':
        bundle['decoder_policy']['relation_conditioning'] = relation_conditioning
    if config.get('speaker_normalization', 'literal') != 'literal':
        bundle['decoder_policy']['speaker_normalization'] = config['speaker_normalization']
    if training_settings['mode'] != 'fullbatch':
        bundle['optimizer_training'] = dict(training_settings, optimizer_updates=optimizer_updates,
            completed_epochs=next_epoch, training_participants=len(groups),
            selected_epoch=best_epoch,
            selected_optimizer_updates=best_epoch*len(reader_participant_batches(len(groups), training_settings, seed, 0)[0]),
            participant_roster_signature=digest(participant_keys), participant_order='seeded_epoch_shuffle',
            order_basis='local_random_seed_epoch_v1', final_batch_divisor='fixed_configured_batch_size')
    bundle['fingerprint'] = reader_fingerprint(bundle)
    backbone.reader = ReaderRuntime(backbone, bundle)
    checkpoint(next_epoch, True, bundle)
    return bundle
