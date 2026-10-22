"""Bounded source-only states reachable from the existing three-text seed."""
from itertools import combinations

from .data import canonical, digest


def teacher_rollout_settings(config):
    settings = config.get('teacher_rollout', {})
    if not isinstance(settings, dict):
        raise ValueError('Teacher rollout must be a configuration dictionary')
    mode = settings.get('mode', 'sampled_prefixes')
    if mode not in ('sampled_prefixes', 'seed_aligned_audio'):
        raise ValueError('Unsupported teacher rollout mode')
    return {'mode': mode}


def aligned_audio_rollout_plan(record, track, initial, options):
    """Cover all nonterminal audio subsets, without adding source text.

    The unchanged seed contains three text observations, hence at most three
    unambiguous aligned audio observations: seven states and twelve transitions.
    Each state's prefix order is deterministic and supplies its timing path.
    """
    if record.get('split') not in ('train', 'dev') or track != 'A+T':
        raise ValueError('Aligned-audio teacher rollouts require source train/dev A+T')
    initial = canonical(initial)
    if len(initial) != 3 or any(view['modality'] != 'T' for view in initial):
        raise ValueError('Aligned-audio teacher rollouts preserve the existing three-text seed')
    seed_ids = tuple(view['id'] for view in initial)
    events = {view['event'] for view in initial}
    audio = [view for view in options if view['modality'] == 'A' and view['event'] in events
             and view['id'] not in seed_ids and view.get('available', True)]
    if len({view['event'] for view in audio}) != len(audio):
        raise ValueError('Source seed event has ambiguous aligned audio observations')
    audio.sort(key=lambda view: (digest((record['family'], record['id'], seed_ids, view['id'])), view['id']))
    plans = []
    for size in range(len(audio)):
        for subset in combinations(audio, size):
            subset_ids = {view['id'] for view in subset}
            ordered = list(subset) + [view for view in audio if view['id'] not in subset_ids]
            plans.append((ordered, size))
    return audio, plans
