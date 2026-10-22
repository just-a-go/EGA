"""One source-fitted Audio logit transport; text-only predictions are unchanged."""
import copy
import math

import numpy as np

from .data import canonical, digest, pool, seed_views
from .metrics import classification, threshold
from . import grpo


MODE = 'source_audio_cut_transport'


def validate_profile(profile):
    if profile is None or profile.get('mode', 'disabled') == 'disabled':
        return {'mode': 'disabled'}
    if profile.get('mode') != MODE or profile.get('version') != 1:
        raise ValueError('Unsupported score calibration profile')
    for key in ('source_text_cut', 'source_audio_cut'):
        value = profile.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError('Score calibration cuts must lie strictly inside (0, 1)')
    offset = math.log(profile['source_text_cut']/(1-profile['source_text_cut'])) - math.log(
        profile['source_audio_cut']/(1-profile['source_audio_cut']))
    if not math.isclose(profile.get('logit_offset', math.inf), offset, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Score calibration offset does not match its source cuts')
    if profile.get('fingerprint') != digest({k: v for k, v in profile.items() if k != 'fingerprint'}):
        raise ValueError('Score calibration fingerprint mismatch')
    return profile


def calibration_settings(config):
    value = config.get('score_calibration', {})
    if not isinstance(value, dict):
        raise ValueError('Controller score_calibration must be a dictionary')
    if value.get('mode', 'disabled') == 'disabled':
        return {'mode': 'disabled'}
    if value.get('mode') != MODE:
        raise ValueError('Unsupported score calibration mode')
    if 'fingerprint' in value:
        return validate_profile(value)
    if set(value) != {'mode'}:
        raise ValueError('Unresolved score calibration must specify only its mode')
    return dict(value)


def probability(raw_p, has_audio, profile=None):
    profile = validate_profile(profile)
    values = np.asarray(raw_p, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Score calibration requires finite probabilities in [0, 1]')
    if not has_audio or profile['mode'] == 'disabled':
        return raw_p
    text, audio = profile['source_text_cut'], profile['source_audio_cut']
    # This odds-ratio form preserves endpoints without log(0), overflow, or clipping.
    left, right = values*text*(1-audio), (1-values)*audio*(1-text)
    result = left/(left+right)
    result = np.where(values == audio, text, result)  # Exactly preserve >=cut ties.
    return float(result) if result.ndim == 0 else result


def fit_source_audio(records, examples, track, plain, text_cut, components, source_signature):
    """Use exactly one pre-existing, deterministically selected aligned A per DEV record."""
    from ..controller import aligned_audio_candidates
    if len(records) != 35 or any(r['split'] != 'dev' for r in records):
        raise ValueError('Audio calibration requires the original 35 source development records')
    lookup = {(r['family'], r['id']): r for r in plain}
    rows, provenance = [], []
    for rec in records:
        initial = canonical(seed_views(rec, track))
        ids = tuple(v['id'] for v in initial)
        if any(v['modality'] == 'A' for v in initial):
            raise ValueError('Audio calibration requires a text-only source seed')
        aligned = aligned_audio_candidates(rec, initial, pool(rec, track))
        if not aligned:
            raise ValueError('Source initial state has no eligible aligned audio')
        matches = [e for e in examples if e['participant'] == rec['family']+':'+rec['participant']
                   and e['state_id'] == (rec['id'], ids) and e['action_id'] == aligned[0]['id']]
        if len(matches) != 1 or not matches[0].get('supervised_valid', True):
            raise ValueError('Source cache is missing its unique initial aligned audio target')
        pair = matches[0]
        seed = lookup[(rec['family'], rec['id'])]
        if (pair['label'] != rec['label'] or seed['label'] != rec['label']
                or not np.isclose(pair['before_probabilities'][0], seed['p'], rtol=0., atol=1e-7)):
            raise ValueError('Audio calibration source label/plain seed changed')
        rows.append({'id': rec['id'], 'family': rec['family'], 'label': rec['label'],
                     'p': float(pair['after_probabilities'][0]), 'status': 'ok'})
        provenance.append({'record': rec['id'], 'action': pair['action_id'],
                           'before_p': float(pair['before_probabilities'][0]),
                           'after_p': rows[-1]['p'], 'label': rec['label']})
    audio_cut = threshold(rows)
    if not 0 < text_cut < 1 or not 0 < audio_cut < 1:
        raise ValueError('Source-selected calibration cuts must lie strictly inside (0, 1)')
    profile = {'mode': MODE, 'version': 1, 'source_text_cut': float(text_cut),
               'source_audio_cut': float(audio_cut),
               'logit_offset': math.log(text_cut/(1-text_cut))-math.log(audio_cut/(1-audio_cut)),
               'selection': 'source_dev_initial_aligned_audio_canonical_macro_f1', 'source_n': 35,
               'source_prediction_signature': grpo.signature(provenance),
               'source_records_signature': source_signature, 'frontend_components': components}
    profile['fingerprint'] = digest(profile)
    validate_profile(profile)
    report = {'profile': profile, 'raw_audio': classification(rows, text_cut),
              'source_audio_cut_metrics': classification(rows, audio_cut), 'rows': provenance,
              'target_adaptation': False, 'new_model_scores': 0}
    return profile, report


def calibrated_record_targets(record, examples, profile, gain_loss):
    """Change only probability units and paired targets, retaining native context/raw triplets."""
    from ..controller import paired_targets, validate_gain_profile
    validate_profile(profile)
    views = {v['id']: v for v in record['views']}
    result = []
    for original in examples:
        if not original.get('supervised_valid', True):
            raise ValueError('Audio score transport requires complete supervised raw triplets')
        validate_gain_profile(original, gain_loss)
        if original.get('score_calibration', {}).get('mode', 'disabled') != 'disabled':
            raise ValueError('Cannot transport an already calibrated source teacher')
        state = [views[key] for key in original['state_id'][1]]
        before_audio = any(v['modality'] == 'A' for v in state)
        after_audio = before_audio or views[original['action_id']]['modality'] == 'A'
        before = list(map(float, probability(original['before_probabilities'], before_audio, profile)))
        after = list(map(float, probability(original['after_probabilities'], after_audio, profile)))
        gain, variance, response = paired_targets(before, after, original['label'], gain_loss)
        example = copy.deepcopy(original)
        example['state'] = np.asarray(original['state'], dtype=np.float32).copy()
        example['state'][0] = before[0]
        example.update(calibrated_before_probabilities=before, calibrated_after_probabilities=after,
                       before_has_audio=before_audio, after_has_audio=after_audio, score_calibration=profile,
                       gain=gain, variance=variance, response=response)
        result.append(example)
    return result
