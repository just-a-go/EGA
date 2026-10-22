"""Derive Controller inputs from verified source teachers without rescoring after states."""
import copy
import time
from pathlib import Path

import numpy as np

from . import grpo
from .data import canonical, seed_views
from ..controller import (STATE_REPRESENTATION_VERSION, encode_state, gain_loss_settings, paired_targets,
                  reader_gate_signature, score_with_reader_gate, validate_gain_profile)
from .models import CONTROLLER_CONTEXT_BASIS


def _probability_valid(prediction):
    return prediction['status'] == 'ok' and prediction.get('probability_valid', True)


def teacher_rebuild_settings(config):
    value = config.get('teacher_input_rebuild', {})
    if not isinstance(value, dict):
        raise ValueError('teacher_input_rebuild must be a configuration dictionary')
    mode = value.get('mode', 'disabled')
    if mode == 'disabled':
        return {'mode': mode}
    if mode != 'native_context_reader_seed':
        raise ValueError('Unsupported teacher input rebuild mode')
    profile = {'mode': mode, 'version': 1, 'context_basis': CONTROLLER_CONTEXT_BASIS}
    if any(value.get(k, v) != v for k, v in profile.items()):
        raise ValueError('Teacher context rebuild version/basis changed')
    if (config.get('state_representation') != 'frozen_discriminator_context'
            or config.get('seed_prediction_anchor') != 'legacy_reader'
            or config.get('grpo', {}).get('enabled', True)):
        raise ValueError('Native context rebuild requires legacy Reader seed and a supervised Controller')
    return profile


def sanctioned_rebuild(original, requested):
    """The one allowed migration; optimizer options do not enter these generation profiles."""
    if requested['teacher_input_rebuild']['mode'] == 'disabled':
        return False
    if (original['state_representation'], original['seed_prediction_anchor']) != (
            'frozen_token_embeddings', 'plain_until_acquisition'):
        return False
    if (requested['state_representation'], requested['seed_prediction_anchor']) != (
            'frozen_discriminator_context', 'legacy_reader'):
        raise ValueError('Unsupported source teacher input migration')
    allowed = {'state_representation', 'seed_prediction_anchor', 'teacher_input_rebuild', 'teacher_augmentation'}
    if {k: v for k, v in original.items() if k not in allowed} != {
            k: v for k, v in requested.items() if k not in allowed}:
        raise ValueError('Input rebuild cannot change source sampling, gate, gain profile, or budget')
    if requested.get('teacher_augmentation', {'mode': 'disabled'})['mode'] != 'disabled':
        raise ValueError('Rebuild requires the already merged base targets, without new augmentation')
    return True


def rebuild_record_targets(backbone, record, track, config, original_pairs, directory, resume, cache_signature):
    """One captured canonical before per state; only initial before C1/C2 are rescored."""
    profile = teacher_rebuild_settings(config)
    if profile['mode'] == 'disabled':
        raise ValueError('Rebuild mode is disabled')
    if record['split'] not in ('train', 'dev'):
        raise ValueError('Teacher rebuilding is source TRAIN/DEV only')
    loss_profile = gain_loss_settings(config)
    gate = config.get('reader_gate')
    views = {v['id']: v for v in record['views']}
    initial_ids = tuple(v['id'] for v in canonical(seed_views(record, track)))
    grouped = {}
    for example in original_pairs:
        if (not example.get('supervised_valid', True) or 'before_probabilities' not in example
                or 'after_probabilities' not in example):
            raise ValueError('Rebuild requires complete valid source raw probability triplets')
        validate_gain_profile(example, loss_profile)
        if (example.get('state_representation') != 'frozen_token_embeddings'
                or example.get('seed_prediction_anchor') != 'plain_until_acquisition'
                or example.get('reader_gate_signature') != reader_gate_signature(gate)
                or int(example['label']) != int(record['label'])):
            raise ValueError('Original source pair input/gate/native-label identity changed')
        state_ids = tuple(example['state_id'][1])
        grouped.setdefault(state_ids, []).append(example)
    output = []
    stats = {'initial_states': 0, 'context_states': len(grouped), 'cache_hits': 0,
             'canonical_before_scores': 0, 'initial_variant_before_scores': 0, 'after_scores': 0}
    if not grouped:
        return output, stats
    if initial_ids not in grouped:
        raise ValueError('Source teacher rebuild requires its exact initial state for budget migration')
    initial_reference = grouped[initial_ids][0]
    seed_seconds = float(initial_reference['seed_seconds'])
    budget = float(config.get('budget', 8))
    initial_fraction = float(initial_reference['state'][2])
    if (not np.isfinite([seed_seconds, budget, initial_fraction]).all()
            or seed_seconds <= 0 or budget <= 0 or not 0 <= initial_fraction <= 1):
        raise ValueError('Invalid source initial timing/budget fraction')
    if any(float(e['seed_seconds']) != seed_seconds or float(e['state'][2]) != initial_fraction
           for e in grouped[initial_ids]):
        raise ValueError('Initial source actions disagree on seed timing or budget')
    old_extra = max(0., budget*seed_seconds*(1-initial_fraction)-seed_seconds)
    budget_migration = None
    directory = Path(directory)
    # Initial timing establishes one record-wide delta before later states are cloned.
    ordered_states = [initial_ids]+[key for key in grouped if key != initial_ids]
    for state_ids in ordered_states:
        examples = grouped[state_ids]
        state = canonical([views[identity] for identity in state_ids])
        if tuple(v['id'] for v in state) != state_ids:
            raise ValueError('Original source state IDs are not canonical')
        initial = state_ids == initial_ids
        stats['initial_states'] += int(initial)
        reference = examples[0]
        if any(not np.array_equal(e['before_probabilities'], reference['before_probabilities'])
               for e in examples):
            raise ValueError('Original actions disagree on their matched before triplet')
        expected = grpo.signature((cache_signature, state_ids, profile))
        path = directory/(grpo.signature(state_ids)[:20]+'.pt')
        cached = grpo.load_checkpoint(path, resume, 'ega-teacher-rebuild-state-v1', expected)
        if cached is None:
            started = time.perf_counter()
            anchor = state[0]['id']
            prediction = score_with_reader_gate(backbone, state, gate, capture_controller_state=True)
            stats['canonical_before_scores'] += 1
            if not _probability_valid(prediction):
                raise ValueError(f'Invalid source canonical rebuild output: {record["id"]}/{state_ids}: '
                                 f'{prediction.get("status")}: {prediction.get("error")}')
            before = list(reference['before_probabilities'])
            if initial:
                before = [float(prediction['p'])]
                for renderer in (1, 2):
                    if prediction['reader_used']:
                        variant = score_with_reader_gate(backbone, state, gate, renderer, anchor,
                                                         evidence=prediction.get('evidence', []))
                    else:
                        variant = getattr(backbone, 'score_unaugmented', backbone.score)(state, renderer, anchor)
                    stats['initial_variant_before_scores'] += 1
                    if not _probability_valid(variant):
                        raise ValueError(f'Invalid initial source renderer {renderer}: {record["id"]}')
                    before.append(float(variant['p']))
            elif abs(float(prediction['p'])-float(before[0])) > 1e-6:
                raise ValueError(f'Frozen noninitial canonical p changed: {record["id"]}/{state_ids}')
            context = prediction['controller_context']
            x = encode_state(state, prediction.get('evidence'), backbone, 'frozen_discriminator_context',
                             controller_context=context)
            cached = {'schema': 'ega-teacher-rebuild-state-v1', 'signature': expected,
                      'x': x, 'controller_context': context, 'before_probabilities': before,
                      'reader_used': prediction['reader_used'],
                      'reader_decision': prediction['reader_decision'],
                      'state_reader_used': bool(prediction.get('evidence')), 'initial': initial,
                      'seconds': time.perf_counter()-started}
            if initial:
                # Accepted Reader screening is an extra runtime forward beyond the
                # cached plain seed. A rebuilt fallback plain forward is not extra.
                new_extra = float(prediction['seconds']) if prediction['reader_used'] else old_extra
                if not np.isfinite(new_extra) or new_extra < 0:
                    raise ValueError('Invalid Reader seed screening time')
                cached['budget_migration'] = {
                    'old_initial_extra_seconds': old_extra, 'new_initial_extra_seconds': new_extra,
                    'initial_extra_delta_seconds': new_extra-old_extra,
                    'initial_reader_used': bool(prediction['reader_used']),
                    'basis': 'seed_reader_extra_delta_v1'}
            grpo.save_checkpoint(path, cached)
        else:
            stats['cache_hits'] += 1
        if initial:
            budget_migration = cached['budget_migration']
            stats['budget_migration'] = budget_migration
        for original in examples:
            example = copy.deepcopy(original)
            example.update(x=cached['x'], controller_context=copy.deepcopy(cached['controller_context']),
                           state_representation='frozen_discriminator_context',
                           state_representation_version=STATE_REPRESENTATION_VERSION,
                           seed_prediction_anchor='legacy_reader', teacher_input_rebuild=profile,
                           reader_used=cached['reader_used'], state_reader_used=cached['state_reader_used'],
                           reader_decision=cached['reader_decision'], budget_migration=budget_migration)
            example['state'] = np.asarray(original['state'], dtype=np.float32).copy()
            denominator = budget*float(original['seed_seconds'])
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError('Invalid cached source seed timing')
            example['state'][2] = max(0., float(original['state'][2])
                                     -budget_migration['initial_extra_delta_seconds']/denominator)
            if initial:
                gain, variance, response = paired_targets(cached['before_probabilities'],
                                                          original['after_probabilities'], record['label'], loss_profile)
                example['state'][0] = cached['before_probabilities'][0]
                example.update(before_probabilities=list(cached['before_probabilities']),
                               gain=gain, variance=variance, response=response)
            # These old reward deltas used the old plain initial answer, and are
            # irrelevant to the supervised Controller. Never invent a new one.
            example.pop('screening_reward_delta', None)
            output.append(example)
    return output, stats
