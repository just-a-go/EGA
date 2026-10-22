"""Source TRAIN ranking references for immutable cached Controller targets."""
import copy
import hashlib
import json

import numpy as np


def ranking_settings(config):
    supplied = config.get('ranking_loss', {})
    if not isinstance(supplied, dict):
        raise ValueError('Controller ranking_loss must be a dictionary')
    settings = {'enabled': False, 'weight': .1, 'temperature': .1} | supplied
    if (not isinstance(settings['enabled'], bool)
            or not np.isfinite([settings['weight'], settings['temperature']]).all()
            or settings['weight'] < 0 or settings['temperature'] <= 0
            or (settings['enabled'] and settings['weight'] == 0)):
        raise ValueError('Invalid Controller ranking-loss settings')
    return settings


def _probabilities(example, side):
    calibrated = 'calibrated_'+side+'_probabilities'
    if calibrated in example:
        values = example[calibrated]
    else:
        profile = example.get('score_calibration')
        if profile is not None and profile.get('mode') != 'disabled':
            raise ValueError('Ranking targets require calibrated renderer probabilities')
        values = example.get(side+'_probabilities')
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Ranking targets require three finite observed probabilities')
    return values


def _training_anchors(train):
    valid = [e for e in train if e.get('supervised_valid', True)]
    labels, minimum = {}, {}
    for example in valid:
        participant, label = str(example['participant']), example['label']
        if label not in (0, 1) or (participant in labels and labels[participant] != label):
            raise ValueError('Ranking TRAIN participant has inconsistent binary labels')
        labels[participant] = int(label)
        record = (participant, str(example['state_id'][0]))
        size = len(example['state_id'][1])
        minimum[record] = min(size, minimum.get(record, size))
    canonical = {}
    for example in valid:
        record = (str(example['participant']), str(example['state_id'][0]))
        if len(example['state_id'][1]) != minimum[record]:
            continue
        state = tuple(example['state_id'][1])
        probability = float(_probabilities(example, 'before')[0])
        if record in canonical and (canonical[record][0] != state or canonical[record][1] != probability):
            raise ValueError('Ranking TRAIN record has inconsistent initial canonical anchors')
        canonical[record] = (state, probability)
    scores = {}
    for (participant, record), (state, probability) in canonical.items():
        scores.setdefault(participant, []).append(probability)
    # One vote per participant, even if a source cohort contains several records.
    anchors = [{'participant': p, 'label': labels[p], 'p': float(np.mean(scores[p])),
                'initial_records': len(scores[p])} for p in sorted(scores)]
    if {a['label'] for a in anchors} != {0, 1}:
        raise ValueError('Source ranking references require both TRAIN classes')
    return anchors


def derive_ranking_targets(train, dev, settings, gain_loss):
    """Return copied targets and a TRAIN-only profile; disabled mode returns exact inputs."""
    if not settings['enabled']:
        return train, dev, {'enabled': False}
    from ..controller import gain_probability
    anchors = _training_anchors(train)
    profile = {'enabled': True, 'source': 'source_train_initial_canonical_participant_votes',
               'settings': copy.deepcopy(settings), 'gain_loss': copy.deepcopy(gain_loss),
               'anchors': anchors, 'class_counts': {str(label): sum(a['label'] == label for a in anchors)
                                                 for label in (0, 1)}}
    profile['fingerprint'] = hashlib.sha256(json.dumps(profile, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    references = {label: np.asarray([a['p'] for a in anchors if a['label'] != label]) for label in (0, 1)}
    def losses(probabilities, label):
        sign = 1. if label == 1 else -1.
        margin = sign*(references[label][None, :]-probabilities[:, None])/settings['temperature']
        pair_loss = (1./(1.+np.exp(-np.clip(margin, -60., 60.)))).mean(1)
        point_loss = (gain_probability(probabilities, gain_loss)-label)**2
        return point_loss+settings['weight']*pair_loss
    def derive(examples):
        output = []
        for original in examples or []:
            example = copy.deepcopy(original)
            if example.get('supervised_valid', True):
                if 'ranking_target_profile' in example:
                    raise ValueError('Ranking targets must derive from the original verified teacher')
                label = example['label']
                if label not in (0, 1):
                    raise ValueError('Ranking target label must be binary')
                before, after = _probabilities(example, 'before'), _probabilities(example, 'after')
                left, right = losses(before, label), losses(after, label)
                differences = left-right
                for key in ('gain', 'variance', 'response'):
                    example['original_'+key] = copy.deepcopy(example[key])
                example.update(gain=float(differences[0]), variance=float(np.var(differences, ddof=1)),
                               response=left-left[0], ranking_target_profile=profile['fingerprint'])
            output.append(example)
        return output
    derived_train, derived_dev = derive(train), derive(dev)
    profile.update(train_targets=sum(e.get('supervised_valid', True) for e in derived_train),
                   dev_targets=sum(e.get('supervised_valid', True) for e in derived_dev))
    return derived_train, derived_dev, profile
