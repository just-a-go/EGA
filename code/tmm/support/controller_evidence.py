"""Add frozen observed Reader records to cached Controller states, without LLM scoring."""
import copy
from pathlib import Path

import numpy as np

from . import grpo
from .data import canonical, seed_views
from ..controller import (COST_BASIS, controller_state_evidence, cost_features, cost_from_features,
                  encode_state, fit_cost)


BUDGET_BASIS = 'observed_reader_initial_current_measured_intermediate_estimated_v1'


def encode_record_states(backbone, record, track, config, pairs, directory, resume, signature):
    """Cache one Reader extraction per original observed state; native scores stay immutable."""
    if record['split'] not in ('train', 'dev'):
        raise ValueError('Controller evidence migration is source TRAIN/DEV only')
    initial = tuple(v['id'] for v in canonical(seed_views(record, track)))
    views = {v['id']: v for v in record['views']}
    grouped = {}
    for pair in pairs:
        if pair.get('state_evidence', 'gated') != 'gated' or not pair.get('controller_context'):
            raise ValueError('Observed Reader migration requires the original gated native-context states')
        grouped.setdefault(tuple(pair['state_id'][1]), []).append(pair)
    if grouped and initial not in grouped:
        raise ValueError('Controller evidence migration requires its source initial state')
    states, hits = {}, 0
    for identifiers, examples in grouped.items():
        observed = canonical([views[identity] for identity in identifiers])
        if tuple(v['id'] for v in observed) != identifiers:
            raise ValueError('Original source state IDs are not canonical')
        reference = examples[0]
        if any(not np.array_equal(e['x'], reference['x'])
               or float(e['state'][0]) != float(reference['state'][0]) for e in examples):
            raise ValueError('Cached actions disagree on their observed state')
        identity = grpo.signature((signature, identifiers, reference['controller_context']))
        path = Path(directory)/(grpo.signature(identifiers)[:20]+'.pt')
        cached = grpo.load_checkpoint(path, resume, 'ega-controller-evidence-state-v1', identity)
        if cached is None:
            prediction = {'reader_used': reference.get('reader_used', False), 'evidence': [],
                          'controller_context': reference['controller_context']}
            evidence, meta = controller_state_evidence(backbone, observed, config.get('reader_gate'),
                prediction=prediction, state_evidence='observed_reader')
            seconds = float(meta['state_reader_seconds'])
            if not np.isfinite(seconds) or seconds < 0:
                raise ValueError('Observed Reader timing must be finite and nonnegative')
            x = encode_state(observed, evidence, backbone, config['state_representation'],
                             controller_context=reference['controller_context'])
            cached = {'schema': 'ega-controller-evidence-state-v1', 'signature': identity,
                      'x': x, 'evidence': evidence, 'meta': meta, 'seconds': seconds,
                      'cost_features': cost_features(observed)}
            grpo.save_checkpoint(path, cached)
        else:
            hits += 1
        states[identifiers] = cached
    timings = [value['cost_features'][1:]+[value['seconds'], COST_BASIS] for value in states.values()]
    return states, timings, {'states': len(states), 'cache_hits': hits, 'new_llm_scores': 0}


def reader_cost_model(timings):
    if not timings:
        raise ValueError('Observed Reader states require measured source TRAIN timings')
    return fit_cost(timings)


def derive_record_pairs(record, track, config, pairs, states, reader_cost):
    """Keep all targets/context/raw outputs; migrate x and runtime Reader budget only."""
    initial = canonical(seed_views(record, track))
    initial_ids = tuple(v['id'] for v in initial)
    views = {v['id']: v for v in record['views']}
    output = []
    for original in pairs:
        identifiers = tuple(original['state_id'][1])
        cached = states[identifiers]
        extra = [views[identity] for identity in identifiers if identity not in initial_ids]
        # Old caches omit acquisition order. Intermediate Reader costs therefore
        # use a deterministic nested observed-state path, not new LLM outputs.
        estimated = sum(cost_from_features(cost_features(canonical(initial+extra[:k])), reader_cost, COST_BASIS)
                        for k in range(1, len(extra)))
        measured = cached['seconds'] + (states[initial_ids]['seconds'] if extra else 0.)
        reader_seconds = measured+estimated
        pair = copy.deepcopy(original)
        pair['x'] = cached['x']
        pair['state'] = np.asarray(original['state'], dtype=np.float32).copy()
        pair['state'][2] = max(0., float(original['state'][2])-reader_seconds/
                              (float(config.get('budget', 8))*float(original['seed_seconds'])))
        pair.update(state_evidence='observed_reader', state_reader_used=bool(cached['evidence']),
                    state_reader_seconds=cached['seconds'], state_reader_meta=copy.deepcopy(cached['meta']),
                    reader_evidence=copy.deepcopy(cached['evidence']),
                    reader_budget_migration={'basis': BUDGET_BASIS,
                        'original_remaining_fraction': float(original['state'][2]),
                        'measured_seconds': measured, 'estimated_intermediate_seconds': estimated,
                        'additional_seconds': reader_seconds})
        output.append(pair)
    return output


def add_reader_timings(original_timings, reader_cost):
    """Original measured LLM/encoding timings remain separately saved by the caller."""
    return [list(row[:2])+[float(row[2])+cost_from_features([1., *row[:2]], reader_cost, row[3]), row[3]]
            for row in original_timings]


def derive_candidate_pairs(record, config, pairs):
    """Add event masks to completed observed-Reader inputs; never re-read media."""
    from .controller_candidates import candidate_conditioning_mode, encode_candidate_mask
    mode = candidate_conditioning_mode(config)
    if record['split'] not in ('train', 'dev') or mode != 'observed_event':
        raise ValueError('Candidate mask migration requires source TRAIN/DEV observed-event inputs')
    lookup = {view['id']: view for view in record['views']}
    output = []
    for original in pairs:
        if (original.get('state_evidence', 'gated') != 'observed_reader'
                or candidate_conditioning_mode(original) != 'none'):
            raise ValueError('Candidate migration requires unchanged observed-Reader teachers without masks')
        identifiers = tuple(original['state_id'][1])
        observed = canonical([lookup[identity] for identity in identifiers])
        if tuple(view['id'] for view in observed) != identifiers:
            raise ValueError('Candidate migration observed view order changed')
        if 'reader_evidence' in original:
            evidence = original['reader_evidence']
        elif 'state_reader_records' in original:
            evidence = original['state_reader_records']
        else:
            raise ValueError('Candidate migration is missing its saved observed Reader records')
        action = lookup[original['action_id']]
        if action['id'] in identifiers:
            raise ValueError('Candidate migration action was already observed')
        mask = encode_candidate_mask(observed, evidence, action,
            original.get('state_representation', 'hash'), mode)
        x = np.asarray(original['x'])
        if x.shape != (len(mask), 71):
            raise ValueError('Candidate mask does not align with the saved observed/Reader/context rows')
        if original.get('state_representation') == 'frozen_discriminator_context':
            if original.get('controller_context', {}).get('observed_view_ids') != list(identifiers):
                raise ValueError('Candidate mask source IDs do not match the saved native context')
        output.append(dict(original, candidate_conditioning=mode, candidate_mask=mask))
    return output


def derive_reader_rows(record, config, pairs, ontology, speaker_normalization='literal'):
    """Replace only saved Reader feature columns; frozen observations stay exact."""
    from .reader_state import encode_reader_record, reader_state_mode
    mode = reader_state_mode(config)
    if record['split'] not in ('train', 'dev'):
        raise ValueError('Reader-row migration is source TRAIN/DEV only')
    lookup = {view['id']: view for view in record['views']}
    output = []
    for original in pairs:
        if original.get('state_evidence', 'gated') != 'observed_reader':
            raise ValueError('Reader-row migration requires completed observed-Reader states')
        identifiers = tuple(original['state_id'][1])
        observed = canonical([lookup[identity] for identity in identifiers])
        if tuple(view['id'] for view in observed) != identifiers:
            raise ValueError('Reader-row source order changed')
        evidence = original.get('reader_evidence', original.get('state_reader_records'))
        if evidence is None:
            raise ValueError('Reader-row migration requires its saved records')
        context_rows = 2 if original.get('state_representation') == 'frozen_discriminator_context' else 0
        x = np.asarray(original['x'])
        if x.shape != (len(observed)+len(evidence)+context_rows, 71):
            raise ValueError('Saved Reader rows do not align with observed/native-context rows')
        observed_lookup = {view['id']: view for view in observed}
        encoded = x.copy()
        for index, evidence_record in enumerate(evidence):
            view = observed_lookup.get(evidence_record['source_id'])
            if view is None:
                raise ValueError('Saved Reader evidence refers to an unobserved source')
            encoded[len(observed)+index, :64] = encode_reader_record(evidence_record, view, ontology, mode,
                                                                   speaker_normalization=speaker_normalization)
        output.append(dict(original, x=encoded, state_evidence_encoding=mode))
    return output
