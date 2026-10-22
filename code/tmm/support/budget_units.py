"""Use one source-fitted second unit for Controller costs and wall-clock budgets."""
import copy
import hashlib
import json
import math
import statistics


def budget_normalization_mode(config):
    mode = config.get('budget_normalization', 'per_seed_wall')
    if mode not in ('per_seed_wall', 'source_train_seed_median'):
        raise ValueError('Unsupported Controller budget normalization')
    return mode


def _positive(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite and positive') from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def _profile(config, profile):
    if (not isinstance(profile, dict) or profile.get('version') != 1
            or profile.get('mode') != 'source_train_seed_median'
            or profile.get('clamped_state_policy') != 'keep_exhausted'):
        raise ValueError('Missing or incompatible source budget profile')
    reference = _positive(profile.get('reference_seconds'), 'Budget reference seconds')
    budget = _positive(profile.get('budget'), 'Budget multiplier')
    signature = profile.get('train_seed_signature')
    if (not isinstance(signature, str) or len(signature) != 64
            or any(c not in '0123456789abcdef' for c in signature)):
        raise ValueError('Invalid source TRAIN timing signature')
    expected = hashlib.sha256(json.dumps({k: v for k, v in profile.items() if k != 'fingerprint'},
                                        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if profile.get('fingerprint') != expected:
        raise ValueError('Source budget profile fingerprint differs')
    if config.get('budget') is not None and float(config['budget']) != budget:
        raise ValueError('Budget multiplier differs from fitted source profile')
    return reference, budget


def fit_budget_profile(train, config):
    """One original seed timing per TRAIN participant, before any target filtering."""
    if budget_normalization_mode(config) == 'per_seed_wall':
        return None
    timings = {}
    for example in train:
        participant = str(example['participant'])
        seconds = _positive(example['seed_seconds'], 'Source seed seconds')
        if participant in timings and timings[participant] != seconds:
            raise ValueError('Source participant has inconsistent seed timings')
        timings[participant] = seconds
    if not timings:
        raise ValueError('Fixed budget units require source TRAIN seed timings')
    seed_values = sorted(timings.items())
    signature = hashlib.sha256(json.dumps(seed_values, separators=(',', ':'),
                                          ensure_ascii=False).encode()).hexdigest()
    profile = {'version': 1, 'mode': 'source_train_seed_median',
               'reference_seconds': float(statistics.median(timings.values())),
               'train_seed_signature': signature,
               'budget': _positive(config.get('budget', 8), 'Budget multiplier'),
               'clamped_state_policy': 'keep_exhausted'}
    profile['fingerprint'] = hashlib.sha256(json.dumps(profile, sort_keys=True,
                                                       separators=(',', ':')).encode()).hexdigest()
    if config.get('budget_profile') is not None and config['budget_profile'] != profile:
        raise ValueError('Source TRAIN timing profile changed')
    return profile


def rebase_budget_examples(examples, config, profile):
    """Preserve raw Teacher inputs; saturated old fractions remain exhausted."""
    original = list(examples or [])
    if budget_normalization_mode(config) == 'per_seed_wall':
        return original
    reference, budget = _profile(config, profile)
    output = []
    for example in original:
        if example.get('budget_rebase') is not None:
            raise ValueError('Controller budget examples were already rebased')
        old_seed = _positive(example['seed_seconds'], 'Source seed seconds')
        fraction = float(example['state'][2])
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('Invalid original Teacher budget fraction')
        lower_bound = budget*old_seed*(1-fraction)
        # float32 fractions have already lost subsecond precision. A zero also
        # loses the entire overrun, so it cannot be promoted into available time.
        spent = (None if fraction == 0 else lower_bound)
        provenance = {'original_remaining_fraction': fraction,
                      'original_seed_seconds': old_seed,
                      'elapsed_seconds': spent, 'spent_lower_bound_seconds': lower_bound,
                      'clamped': fraction == 0,
                      'basis': 'recover_positive_fraction_keep_exhausted_v1'}
        derived = dict(example, state=copy.deepcopy(example['state']),
                       budget_normalization='source_train_seed_median',
                       budget_profile=copy.deepcopy(profile), budget_rebase=provenance)
        derived['state'][2] = (0. if spent is None else
                               max(0., 1-spent/(budget*reference)))
        if 'cost' in example:
            derived['cost'] = float(example['cost'])*old_seed/reference
        output.append(derived)
    return output


def budget_reference(policy, observed_seed_seconds):
    """Actual seed/elapsed timing is recorded separately from this fixed unit."""
    seconds = float(observed_seed_seconds)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Observed seed seconds must be finite and nonnegative')
    if budget_normalization_mode(policy) == 'per_seed_wall':
        return max(1e-6, seconds)
    reference, _ = _profile(policy, policy.get('budget_profile'))
    return reference
