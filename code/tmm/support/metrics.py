"""Participant-paired endpoints used by Tables I--V."""
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score


def threshold(rows):
    if not rows:
        raise ValueError('Development predictions required')
    candidates = sorted({0., .5, 1.0000001, *[r['p'] for r in rows]})
    return max(candidates, key=lambda t: (classification(rows, t)['macro_f1'], -abs(t-.5), -t))


def classification(rows, cut):
    if not rows:
        raise ValueError('Cannot score an empty cohort')
    y = np.array([r['label'] for r in rows])
    p = np.array([r['p'] for r in rows])
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid probability')
    pred = np.array([r['p'] >= r.get('threshold', cut) for r in rows])
    return {'n': len(rows), 'macro_f1': float(f1_score(y, pred, average='macro', labels=[0, 1], zero_division=0)),
            'auroc': float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            'brier': float(np.mean((p-y)**2)), 'mcc': float(matthews_corrcoef(y, pred)),
            'fail_pct': 100 * sum(r['status'] != 'ok' for r in rows)/len(rows)}


def paired(before, after, cut):
    a = {r['id']: r for r in before}
    b = {r['id']: r for r in after}
    if len(a) != len(before) or len(b) != len(after) or a.keys() != b.keys():
        raise ValueError('Paired predictions require identical unique sample IDs')
    correct = harm = wrong = right = flips = 0
    delta = []
    for key, x in a.items():
        z = b[key]
        if x['label'] != z['label']:
            raise ValueError('Label changed within pair')
        t = x.get('threshold', cut)
        ca, cb = (x['p'] >= t) == x['label'], (z['p'] >= t) == x['label']
        wrong += not ca
        right += ca
        correct += not ca and cb
        harm += ca and not cb
        flips += ca != cb
        delta.append((z['p']-z['label'])**2 - (x['p']-x['label'])**2)
    n = len(a)
    return {'n': n, 'correction_pct': 100*correct/wrong if wrong else None,
            'harm_pct': 100*harm/right if right else None, 'initially_wrong': wrong,
            'initially_correct': right, 'net_gain_pct': 100*(correct-harm)/n,
            'corrected': correct, 'harmed': harm, 'flipped': flips,
            'unconditional_correction_pct': 100*correct/n,
            'unconditional_harm_pct': 100*harm/n,
            'flip_pct': 100*flips/n, 'delta_brier': float(np.mean(delta))}


def paired_bootstrap(before, after, cut, repeats=2000, seed=27):
    """Intervals for paired endpoints, resampling whole participants together."""
    paired(before, after, cut)  # Enforce unique aligned identities and unchanged labels.
    mapping = {r['id']: r for r in after}
    groups = {}
    vectors = []
    for i, x in enumerate(before):
        z = mapping[x['id']]
        t = x.get('threshold', cut)
        ca, cb = (x['p'] >= t) == x['label'], (z['p'] >= t) == x['label']
        vectors.append([not ca and cb, ca and not cb, not ca, ca, ca != cb,
                        (z['p']-z['label'])**2-(x['p']-x['label'])**2])
        groups.setdefault((x['family'], x['participant']), []).append(i)
    values, rng = {}, np.random.default_rng(seed)
    groups, vectors = list(groups.values()), np.asarray(vectors, dtype=float)
    for _ in range(repeats):
        indices = [i for g in rng.integers(len(groups), size=len(groups)) for i in groups[g]]
        corrected, harmed, wrong, right, flips, delta = vectors[indices].sum(0)
        n = len(indices)
        scores = {'correction_pct': 100*corrected/wrong if wrong else None,
                  'harm_pct': 100*harmed/right if right else None,
                  'unconditional_correction_pct': 100*corrected/n,
                  'unconditional_harm_pct': 100*harmed/n,
                  'net_gain_pct': 100*(corrected-harmed)/n,
                  'flip_pct': 100*flips/n, 'delta_brier': delta/n}
        for key, value in scores.items():
            if value is not None:
                values.setdefault(key, []).append(value)
    keys = ('correction_pct', 'harm_pct', 'unconditional_correction_pct',
            'unconditional_harm_pct', 'net_gain_pct', 'flip_pct', 'delta_brier')
    return {key: np.quantile(values[key], [.025, .975]).tolist() if values.get(key) else None
            for key in keys}


def bootstrap(before, after=None, cut=.5, repeats=2000, seed=27):
    groups = {}
    for i, row in enumerate(before):
        groups.setdefault((row['family'], row['participant']), []).append(i)
    keys = list(groups)
    if after is not None:
        mapping = {r['id']: r for r in after}
        if len(mapping) != len(after) or set(mapping) != {r['id'] for r in before}:
            raise ValueError('Bootstrap pairs do not align')
        after = [mapping[r['id']] for r in before]
    rng, values = np.random.default_rng(seed), {}
    for _ in range(repeats):
        indices = [i for g in rng.integers(len(keys), size=len(keys)) for i in groups[keys[g]]]
        score = classification([before[i] for i in indices], cut)
        if after is not None:
            post = classification([after[i] for i in indices], cut)
            score = {k: post[k]-v if v is not None and post[k] is not None else None
                     for k, v in score.items() if k != 'n'}
        for k, v in score.items():
            if k != 'n' and v is not None:
                values.setdefault(k, []).append(v)
    return {k: np.quantile(v, [.025, .975]).tolist() for k, v in values.items()}
