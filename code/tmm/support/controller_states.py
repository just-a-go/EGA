"""Source state eligibility from observation identities; no labels or media reads."""


def state_filter_mode(config):
    mode = config.get('state_filter', 'none')
    if mode not in ('none', 'seed_aligned_audio'):
        raise ValueError('Unsupported Controller state filter')
    return mode


def mark_record_states(record, pairs, mode='seed_aligned_audio'):
    """Mark original record pairs before any candidate/action filtering."""
    mode = state_filter_mode({'state_filter': mode})
    original = pairs
    pairs = list(pairs or [])
    if mode == 'none' or not pairs:
        return original, _statistics(pairs, pairs, mode)
    lookup = {view['id']: view for view in record['views']}
    if len(lookup) != len(record['views']):
        raise ValueError('Controller state filter requires unique source view IDs')
    states = set()
    for pair in pairs:
        record_id, observed = pair['state_id']
        observed = tuple(observed)
        if str(record_id) != str(record['id']) or len(set(observed)) != len(observed):
            raise ValueError('Controller state identity does not match its source record')
        if not observed or any(view_id not in lookup for view_id in observed):
            raise ValueError('Controller state contains missing source view IDs')
        states.add(observed)
    minimum = min(map(len, states))
    seeds = [ids for ids in states if len(ids) == minimum]
    if len(seeds) != 1:
        raise ValueError('Controller state filter requires one consistent initial seed state')
    seed = seeds[0]
    seed_set = set(seed)
    text_events = {lookup[view_id]['event'] for view_id in seed
                   if lookup[view_id]['modality'] == 'T'}
    if not text_events:
        raise ValueError('Aligned-audio state filter requires seed text observations')
    marked = []
    for pair in pairs:
        observed = tuple(pair['state_id'][1])
        extras = [lookup[view_id] for view_id in observed if view_id not in seed_set]
        reason = ('missing_seed' if not seed_set.issubset(observed) else
                  'additional_non_audio' if any(view['modality'] != 'A' for view in extras) else
                  'unaligned_audio' if any(view['event'] not in text_events for view in extras) else
                  'reachable')
        marked.append(dict(pair, state_filter=mode, state_filter_eligible=reason == 'reachable',
                           state_filter_reason=reason, state_filter_seed_ids=list(seed)))
    return marked, _statistics(marked, [p for p in marked if p['state_filter_eligible']], mode)


def filter_controller_states(examples, mode='none'):
    mode = state_filter_mode({'state_filter': mode})
    original = list(examples or [])
    if mode == 'none':
        return original, _statistics(original, original, mode)
    seeds = {}
    for pair in original:
        if (state_filter_mode(pair) != mode or type(pair.get('state_filter_eligible')) is not bool
                or not pair.get('state_filter_seed_ids')):
            raise ValueError('Controller state filter requires record-aware source eligibility markers')
        key = (pair['participant'], str(pair['state_id'][0]))
        seed = tuple(pair['state_filter_seed_ids'])
        if key in seeds and seeds[key] != seed:
            raise ValueError('Controller state filter seed changed within a source record')
        seeds[key] = seed
    retained = [pair for pair in original if pair['state_filter_eligible']]
    return retained, _statistics(original, retained, mode)


def _statistics(original, retained, mode):
    def states(pairs):
        return {(pair['participant'], str(pair['state_id'][0]), tuple(pair['state_id'][1]))
                for pair in pairs}
    return {'mode': mode, 'original_pairs': len(original), 'retained_pairs': len(retained),
            'removed_pairs': len(original)-len(retained), 'original_states': len(states(original)),
            'retained_states': len(states(retained)),
            'removed_states': len(states(original))-len(states(retained))}
