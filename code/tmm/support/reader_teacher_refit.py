"""Reader-only derivation of verified, complete native teacher records.

The caller owns checkpoint/cache verification. This helper never scores the LLM
and leaves the source reference probabilities, targets and costs unchanged.
"""
import copy

import numpy as np

from . import grpo
from .controller_candidates import candidate_conditioning_mode, encode_candidate_mask
from .controller_rollouts import aligned_audio_rollout_plan, teacher_rollout_settings
from .data import canonical, metadata, pool, seed_views
from ..controller import (COST_BASIS, controller_context_rows, controller_inputs,
                  controller_state_evidence, gain_loss_settings, reader_gate_signature)
from .reader_state import (encode_reader_record, reader_state_mode, reader_state_profile,
                           RELATION_PROBABILITY_BASIS)


SCHEMA = 'ega-reader-teacher-refit-v1'


def record_signature(record):
    """Use the existing teacher record identity, excluding indexing diagnostics."""
    stable = {k: v for k, v in record.items() if k not in ('index_seconds', 'views')}
    stable['views'] = [{k: v for k, v in view.items() if k != 'reader_positions'}
                       for view in record['views']]
    return grpo.signature([stable])


def native_identity(backbone, track='A+T'):
    bundle = getattr(backbone, 'discriminator_bundle', None)
    fingerprint = bundle.get('fingerprint') if isinstance(bundle, dict) else None
    if not fingerprint:
        raise ValueError('Reader teacher reuse requires a verified frozen discriminator')
    return {'backbone_spec': copy.deepcopy(backbone.spec),
            'protocol': copy.deepcopy(backbone.protocol), 'instruction': backbone.instruction,
            'discriminator_fingerprint': fingerprint, 'track': track, 'renderers': [0, 1, 2]}


def _unavailable(reason, timings, **details):
    return None, timings, {'schema': SCHEMA, 'reusable': False, 'reason': reason, **details}


def _raw_probabilities(example):
    for key in ('before_probabilities', 'after_probabilities'):
        values = np.asarray(example.get(key), dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all() or ((values < 0.) | (values > 1.)).any():
            return False
    return bool(example.get('supervised_valid', True))


def derive_refit_reader_teacher(backbone, record, raw_pairs, raw_timings, config, verified_origin):
    """Return new Reader inputs, the *same* timings, and separate derivation proof.

    ``config`` is the resolved Controller teacher configuration with native score
    calibration disabled. A non-reusable gate/coverage returns ``None`` as the
    first item; an inconsistent native identity raises instead of substituting
    another source. Only the seven nonterminal observed states are read.
    """
    origin = verified_origin
    if origin.get('schema') != 'ega-plain-teacher-origin-v1':
        raise ValueError('Reader teacher reuse requires a verified plain teacher origin')
    actual_native = native_identity(backbone)
    if origin.get('native_identity') != actual_native:
        raise ValueError('Plain teacher native model, instruction, renderer or discriminator identity changed')
    if origin.get('record_signature') != record_signature(record):
        raise ValueError('Plain teacher native source record identity changed')
    old_reader = origin.get('source_reader_fingerprint')
    new_reader = getattr(getattr(backbone, 'reader', None), 'fingerprint', None)
    if not old_reader or not new_reader or not origin.get('source_record_cache_sha256'):
        raise ValueError('Plain teacher origin is missing verified Reader or raw-cache identity')
    old_gate, gate = origin.get('source_reader_gate'), config.get('reader_gate')
    if origin.get('destination_reader_gate') != gate:
        raise ValueError('Plain teacher destination gate identity changed')
    if (not isinstance(old_gate, dict) or not isinstance(gate, dict)
            or old_gate.get('accept') is not False or gate.get('accept') is not False):
        return _unavailable('classifier_reader_gate_not_rejected_on_both_sides', raw_timings)
    old_config = origin.get('source_teacher_config')
    if not isinstance(old_config, dict):
        raise ValueError('Plain teacher origin is missing its original teacher configuration')
    representation, anchor = controller_inputs(config)
    if (representation != 'frozen_discriminator_context'
            or config.get('state_evidence') != 'observed_reader'
            or teacher_rollout_settings(config) != {'mode': 'seed_aligned_audio'}):
        return _unavailable('unsupported_reader_refit_state_contract', raw_timings)
    if (controller_inputs(old_config) != (representation, anchor)
            or teacher_rollout_settings(old_config) != teacher_rollout_settings(config)
            or gain_loss_settings(old_config) != gain_loss_settings(config)
            or old_config.get('score_calibration', {}).get('mode', 'disabled') != 'disabled'
            or config.get('score_calibration', {}).get('mode', 'disabled') != 'disabled'
            or old_config.get('reader_gate') != old_gate):
        raise ValueError('Plain teacher native score, target loss, anchor or rollout contract changed')
    if record.get('split') not in ('train', 'dev'):
        raise ValueError('Reader teacher derivation only permits native source TRAIN/DEV records')
    initial = seed_views(record, 'A+T')
    options = [view for view in pool(record, 'A+T') if view['id'] not in {v['id'] for v in initial}]
    audio, plans = aligned_audio_rollout_plan(record, 'A+T', initial, options)
    if len(audio) != 3:
        return _unavailable('source_aligned_audio_coverage_incomplete', raw_timings)
    expected = {}
    for ordered, count in plans:
        views = canonical(initial + ordered[:count])
        ids = tuple(view['id'] for view in views)
        expected[ids] = (views, {view['id'] for view in ordered[count:]})
    expected_edges = {(ids, action) for ids, (_, actions) in expected.items() for action in actions}
    groups, found_edges = {}, set()
    lookup = {view['id']: view for view in record['views']}
    for example in raw_pairs:
        state_id = example.get('state_id')
        if not isinstance(state_id, (list, tuple)) or len(state_id) != 2 or state_id[0] != record['id']:
            raise ValueError('Plain teacher state belongs to another native source record')
        ids, action = tuple(state_id[1]), example.get('action_id')
        if (ids, action) not in expected_edges:
            return _unavailable('plain_teacher_rollout_has_unmatched_state_or_transition', raw_timings)
        if (example.get('reader_fingerprint') != old_reader
                or example.get('discriminator_fingerprint') != actual_native['discriminator_fingerprint']
                or example.get('reader_gate_signature') != reader_gate_signature(old_gate)
                or example.get('reader_used') is not False
                or example.get('cost_basis') != COST_BASIS
                or example.get('seed_prediction_anchor', 'legacy_reader') != anchor
                or example.get('state_representation') != representation
                or example.get('teacher_rollout') != {'mode': 'seed_aligned_audio'}
                or gain_loss_settings({'gain_loss': example.get('gain_loss', {'mode': 'raw_brier'})})
                    != gain_loss_settings(config)):
            raise ValueError('Plain teacher pair has inconsistent native/source Reader provenance')
        if (not _raw_probabilities(example)
                or example.get('score_calibration', {}).get('mode', 'disabled') != 'disabled'):
            return _unavailable('plain_teacher_native_probabilities_incomplete', raw_timings)
        if example.get('label') != record['label']:
            raise ValueError('Plain teacher native source label identity changed')
        state = np.asarray(example.get('state'), dtype=float)
        if (state.shape != (3,) or not np.isfinite(state).all()
                or abs(state[0]-float(example['before_probabilities'][0])) > 1e-6):
            raise ValueError('Plain teacher state probability no longer represents native C0')
        groups.setdefault(ids, []).append(example)
        found_edges.add((ids, action))
    if set(groups) != set(expected) or found_edges != expected_edges:
        return _unavailable('plain_teacher_7_states_12_transitions_incomplete', raw_timings,
                            states=len(groups), transitions=len(found_edges))

    mode = reader_state_mode(config)
    encoding_profile = reader_state_profile(config)
    if mode == 'structured_soft':
        if reader_state_mode(old_config) != 'structured':
            raise ValueError('Soft Reader teacher derivation requires the verified structured plain-score origin')
        if getattr(backbone.reader, 'include_relation_probabilities', False) is not True:
            raise ValueError('Soft Reader teacher derivation requires the declared probability payload runtime')
    payload_policy = {'include_relation_probabilities': mode == 'structured_soft',
                      'frozen_reader_fingerprint': new_reader}
    if mode == 'structured_soft':
        payload_policy['relation_probability_basis'] = RELATION_PROBABILITY_BASIS
    conditioning = candidate_conditioning_mode(config)
    ontology = backbone.reader.config.get('ontology')
    updated, reads = {}, []
    for ids, examples in groups.items():
        views = copy.deepcopy(expected[ids][0])
        context = examples[0].get('controller_context')
        try:
            context_rows = controller_context_rows(views, context)
        except RuntimeError:
            return _unavailable('plain_teacher_prompt_context_incomplete', raw_timings)
        original_rows = None
        for example in examples:
            old_evidence = example.get('state_reader_records', example.get('reader_evidence', [])) or []
            x = np.asarray(example.get('x'), dtype=np.float32)
            if (x.shape != (len(views)+len(old_evidence)+2, 71) or not np.isfinite(x).all()
                    or grpo.signature(example.get('controller_context')) != grpo.signature(context)
                    or not np.array_equal(x[-2:], context_rows)):
                return _unavailable('plain_teacher_state_row_or_context_coverage_incomplete', raw_timings)
            observations = x[:len(views)]
            if original_rows is not None and not np.array_equal(observations, original_rows):
                raise ValueError('Plain teacher observed rows disagree within the same native state')
            original_rows = observations.copy()
        evidence, read = controller_state_evidence(backbone, views, gate, state_evidence='observed_reader')
        if read['state_reader_decision'] == 'reader_input_failure_fallback':
            return _unavailable('refit_reader_observed_input_unavailable', raw_timings)
        observed = {view['id']: view for view in views}
        reader_rows = [np.concatenate([encode_reader_record(item, observed[item['source_id']], ontology, mode,
                        speaker_normalization=backbone.reader.config.get('speaker_normalization', 'literal')),
                                       metadata(observed[item['source_id']], views)]) for item in evidence]
        rows = np.concatenate([original_rows, np.asarray(reader_rows, dtype=np.float32).reshape(-1, 71),
                               context_rows], axis=0)
        updated[ids] = (views, evidence, rows, read)
        reads.append({'observed_view_ids': list(ids), 'records': len(evidence),
                      'new_reader_read_seconds': read['state_reader_seconds'],
                      'decision': read['state_reader_decision']})

    derived = []
    for original in raw_pairs:
        example = copy.deepcopy(original)
        ids = tuple(original['state_id'][1])
        views, evidence, rows, read = updated[ids]
        example.update(x=rows.copy(), state_reader_records=copy.deepcopy(evidence),
                       reader_evidence=copy.deepcopy(evidence), state_reader_used=bool(evidence),
                       state_reader_decision=read['state_reader_decision'],
                       state_reader_observed_view_ids=list(ids), state_reader_fingerprint=new_reader,
                       reader_fingerprint=new_reader, reader_gate_signature=reader_gate_signature(gate),
                       state_evidence_encoding=mode, candidate_conditioning=conditioning,
                       state_evidence_encoding_profile=copy.deepcopy(encoding_profile),
                       reader_payload_policy=copy.deepcopy(payload_policy),
                       source_reader_fingerprint=old_reader,
                       new_reader_read_seconds=read['state_reader_seconds'])
        mask = encode_candidate_mask(views, evidence, lookup[example['action_id']], representation, conditioning)
        if mask is None:
            example.pop('candidate_mask', None)
        else:
            example['candidate_mask'] = mask
        # state_reader_seconds is the *original source reference* cost. The new
        # measured read time above is diagnostic and never changes state[2].
        derived.append(example)
    proof = {'schema': SCHEMA, 'reusable': True, 'source_record_signature': record_signature(record),
             'source_raw_signature': grpo.signature((raw_pairs, raw_timings)),
             'source_record_cache_sha256': origin['source_record_cache_sha256'],
             'source_reader_fingerprint': old_reader, 'derived_reader_fingerprint': new_reader,
             'native_identity': actual_native, 'source_reader_gate_signature': reader_gate_signature(old_gate),
             'derived_reader_gate_signature': reader_gate_signature(gate),
             'state_evidence_encoding': mode, 'candidate_conditioning': conditioning,
             'reader_state_profile': encoding_profile, 'reader_payload_policy': payload_policy,
             'native_scoring_contract': {
                 'mode': 'plain_native_score_both_reader_gates_rejected_v1',
                 'relation_payload_enters_native_prompt': False,
                 'native_probability_triplets': 'unchanged_original_source_reference',
                 'new_llm_calls': 0},
             'states': len(groups), 'transitions': len(found_edges), 'pairs': len(derived),
             'cost_policy': 'fixed_original_source_reference_v1', 'new_read_time_is_diagnostic_only': True,
             'reader_reads': reads}
    stable = [{k: v for k, v in example.items() if k != 'new_reader_read_seconds'} for example in derived]
    proof['derived_signature'] = grpo.signature((proof['source_raw_signature'], actual_native, new_reader,
                                                 gate, encoding_profile, payload_policy, conditioning,
                                                 proof['native_scoring_contract'], stable))
    return derived, raw_timings, proof
