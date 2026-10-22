"""Bind a candidate to already-observed text and Reader rows, without reading its content."""
import numpy as np

from .data import canonical


def candidate_conditioning_mode(config):
    mode = config.get('candidate_conditioning', 'none')
    if mode not in ('none', 'observed_event'):
        raise ValueError('Unsupported Controller candidate conditioning')
    return mode


def action_filter_mode(config):
    mode = config.get('action_filter', 'none')
    if mode not in ('none', 'aligned_audio'):
        raise ValueError('Unsupported Controller action filter')
    return mode


def candidate_action_allowed(candidate, mode='none'):
    mode = action_filter_mode({'action_filter': mode})
    if mode == 'none':
        return True
    values = np.asarray(candidate, dtype=np.float32)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError('Controller action filtering requires finite seven-value metadata')
    return bool(values[1] == 1. and values[6] == 1.)


def encode_candidate_mask(views, evidence, action, state_representation='hash', candidate_conditioning='none'):
    """Use exactly encode_state's canonical-view, Reader-record, then context row order."""
    mode = candidate_conditioning_mode({'candidate_conditioning': candidate_conditioning})
    if mode == 'none':
        return None
    observed = canonical(views)
    lookup = {view['id']: view for view in observed}
    if len(lookup) != len(observed) or not action.get('event'):
        raise ValueError('Candidate conditioning requires unique observed IDs and an event')
    event = action['event']
    mask = [view['modality'] == 'T' and view['event'] == event for view in observed]
    for record in evidence or []:
        if record['source_id'] not in lookup:
            raise ValueError('Candidate Reader row refers to an unobserved source')
        mask.append(lookup[record['source_id']]['event'] == event)
    if state_representation == 'frozen_discriminator_context':
        mask.extend([False, False])
    return np.asarray(mask, dtype=np.float32)


def validate_candidate_mask(example, mode):
    if candidate_conditioning_mode(example) != mode:
        raise ValueError('Teacher candidate conditioning changed; derive matching source masks')
    if mode == 'none':
        return
    mask = np.asarray(example.get('candidate_mask'), dtype=np.float32)
    if (mask.shape != (len(example['x']),) or not np.isfinite(mask).all()
            or not np.isin(mask, [0., 1.]).all()):
        raise ValueError('Candidate mask must contain one binary value per observed row')
    if example.get('state_representation') == 'frozen_discriminator_context' and mask[-2:].any():
        raise ValueError('Global native context cannot enter the candidate event mask')
