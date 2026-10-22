"""Training-only standardization and frozen-backbone feature adapters."""
import copy
import random

import numpy as np
import torch
from torch import nn

from . import grpo
from .data import ObservationError, seed_views
from .models import feature_vector


def _signature_records(records):
    # Loading and reader-position profiling both update this runtime-only field.
    # Observation intervals and source duration remain compatibility inputs.
    return [{**{k: v for k, v in rec.items() if k not in ('index_seconds', 'views')},
             'views': [{k: v for k, v in view.items() if k != 'reader_positions'}
                       for view in rec['views']]} for rec in records]


class FeatureAdapter(nn.Module):
    def __init__(self, stats, hidden_size, device, dtype):
        super().__init__()
        self.names = sorted(stats)
        self.maps = nn.ModuleDict()
        for name in self.names:
            mean, scale = stats[name]
            self.register_buffer(name+'_mean', torch.tensor(mean, dtype=torch.float32))
            self.register_buffer(name+'_scale', torch.tensor(scale, dtype=torch.float32))
            self.maps[name] = nn.Sequential(nn.Linear(len(mean), 256), nn.ReLU(), nn.Linear(256, hidden_size*8))
        self.hidden_size, self.output_dtype = hidden_size, dtype
        self.to(device)

    def forward(self, values):
        tokens = []
        for name in self.names:
            if name not in values:
                continue
            mean, scale = getattr(self, name+'_mean'), getattr(self, name+'_scale')
            x = torch.tensor(values[name], device=mean.device, dtype=torch.float32)
            tokens.append(self.maps[name]((x-mean)/scale).reshape(1, 8, self.hidden_size))
        if not tokens:
            raise ValueError('No feature modality overlaps the fitted adapter')
        return torch.cat(tokens, dim=1).to(self.output_dtype)


def fit_adapter(backbone, train, dev, config, selected=None,
                checkpoint_path=None, resume=False):
    """Fit an adapter, resuming only from completed epoch boundaries."""
    failures, failure_index = [], {}
    def exclude(stage, rec, status, error, answer=''):
        key = (stage, rec.get('family'), rec.get('id'), status, str(error))
        if key in failure_index:
            failure_index[key]['count'] += 1
            return
        item = {'stage': stage, 'family': rec.get('family'), 'record': rec.get('id'),
                'status': status, 'error': str(error), 'answer': str(answer)[:200], 'count': 1}
        failures.append(item)
        failure_index[key] = item
        print(f'adapter_excluded: {item}', flush=True)

    statistics = {}
    eligible = []
    for rec in train:
        try:
            values = [feature_vector(view) for view in seed_views(rec, 'F')]
        except (ObservationError, FileNotFoundError) as exc:
            exclude('statistics', rec, 'input_failure', exc)
            continue
        eligible.append(rec)
        for row in values:
            for name, value in row.items():
                statistics.setdefault(name, []).append(value)
    if not eligible or not statistics:
        raise ValueError('Feature adapter has no usable training observations')
    stats = {k: (np.mean(v, axis=0), np.maximum(np.std(v, axis=0), 1e-6)) for k, v in statistics.items()}
    size = backbone.model.get_input_embeddings().weight.shape[1]
    decays = [selected['weight_decay']] if selected else list(config['weight_decay'])
    epochs = int(selected['epochs'] if selected else config['epochs'])
    if not decays or epochs < 1:
        raise ValueError('Feature adapter requires weight decays and positive epochs')
    expected = grpo.signature((config, selected, stats, _signature_records(train), _signature_records(dev),
                               size, str(backbone.dtype),
                               getattr(backbone, 'spec', {}), getattr(backbone, 'protocol', {})))
    restored = grpo.load_checkpoint(checkpoint_path, resume, 'ega-adapter-training-v1', expected)
    best, trials, history, steps = None, [], [], 0
    start_decay = 0
    if restored is not None:
        best, trials, history, steps = (restored[k] for k in ('best', 'trials', 'history', 'optimizer_steps'))
        failures = restored['excluded']
        failure_index = {(x['stage'], x['family'], x['record'], x['status'], x['error']): x for x in failures}
        start_decay = restored['decay_index'] + int(restored['trial_complete'])
        grpo.restore_rng(restored['rng'])
        print(f'adapter_resume: next_decay={start_decay} epoch={restored["epoch"]} complete={restored["complete"]}', flush=True)
    else:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
    groups = {}
    for rec in eligible:
        groups.setdefault((rec.get('family'), rec['participant']), []).append(rec)

    def finish(adapter):
        adapter.load_state_dict(best['state_dict'])
        adapter.eval().requires_grad_(False)
        backbone.adapter = adapter
        return {'stats': stats, **best, 'trials': trials, 'history': history,
                'optimizer_steps': steps, 'excluded': failures, 'resumed': restored is not None}

    if restored is not None and restored['complete']:
        adapter = FeatureAdapter(stats, size, backbone.device, backbone.dtype)
        result = finish(adapter)
        grpo.restore_rng(restored['rng'])
        return result

    for decay_index in range(start_decay, len(decays)):
        decay = decays[decay_index]
        torch.manual_seed(config['seed'])
        adapter = FeatureAdapter(stats, size, backbone.device, backbone.dtype)
        backbone.adapter = adapter
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=config['learning_rate'], weight_decay=decay)
        score_best, stale, epoch_best, saved, start_epoch = float('inf'), 0, 0, None, 0
        if restored is not None and decay_index == restored['decay_index'] and not restored['trial_complete']:
            adapter.load_state_dict(restored['adapter'])
            optimizer.load_state_dict(restored['optimizer'])
            score_best, stale, epoch_best, saved, start_epoch = (
                restored[k] for k in ('score_best', 'stale', 'epoch_best', 'best_trial', 'epoch'))
            grpo.restore_rng(restored['rng'])
        for epoch in range(start_epoch, epochs):
            adapter.train()
            updates = 0
            for records in groups.values():
                optimizer.zero_grad()
                usable = 0
                for rec in records:
                    try:
                        batch, _ = backbone.label_batch(seed_views(rec, 'F'))
                    except (ObservationError, FileNotFoundError) as exc:
                        exclude('train', rec, 'input_failure', exc)
                        continue
                    logits = backbone.model(**batch, use_cache=False).logits[0, -1, backbone.label_ids].float()
                    value = (logits.softmax(-1)[1]-rec['label'])**2
                    if not torch.isfinite(value):
                        raise RuntimeError('Non-finite feature adapter training loss')
                    value.backward()
                    usable += 1
                if not usable:
                    continue
                # Preserve equal participant weighting if some inputs failed.
                for parameter in adapter.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(usable)
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in adapter.parameters()):
                    raise RuntimeError('Non-finite feature adapter optimizer update')
                steps += 1
                updates += 1
            if not updates:
                raise ValueError('Feature adapter epoch has no usable training inputs')
            adapter.eval()
            scores = []
            for rec in (dev or train):
                try:
                    prediction = backbone.score(seed_views(rec, 'F'))
                except (ObservationError, FileNotFoundError) as exc:
                    exclude('dev', rec, 'input_failure', exc)
                    continue
                status = prediction.get('status', 'ok')
                if status not in ('ok', 'input_failure', 'output_failure'):
                    raise RuntimeError(f'Unexpected feature validation status: {status}')
                if status != 'ok' or not prediction.get('probability_valid', status == 'ok'):
                    exclude('dev', rec, status, prediction.get('error') or 'Prediction has no valid probability',
                            prediction.get('answer', ''))
                    continue
                p = float(prediction['p'])
                if not np.isfinite(p) or not 0 <= p <= 1:
                    raise RuntimeError('Invalid feature adapter validation probability')
                scores.append((p-rec['label'])**2)
            if not scores:
                raise ValueError('Feature adapter has no valid validation predictions; fallback scores cannot select a checkpoint')
            score = float(np.mean(scores))
            if selected is not None or score < score_best:
                score_best, stale, saved = score, 0, copy.deepcopy(adapter.state_dict())
                epoch_best = epoch+1
            else:
                stale += 1
            done = (epoch+1 == epochs or (selected is None and stale >= config['patience']))
            history.append({'weight_decay': decay, 'epoch': epoch+1, 'dev_brier': score,
                            'valid_predictions': len(scores), 'excluded_predictions': len(dev or train)-len(scores)})
            print(f'adapter_epoch: decay={decay} epoch={epoch+1} dev_brier={score:.6f} valid={len(scores)}', flush=True)
            if done:
                trial = {'weight_decay': decay, 'dev_brier': score_best, 'epochs': epoch_best}
                trials.append(trial)
                if best is None or score_best < best['dev_brier']:
                    best = {**trial, 'state_dict': saved}
            if checkpoint_path is not None:
                grpo.save_checkpoint(checkpoint_path, {
                    'schema': 'ega-adapter-training-v1', 'signature': expected,
                    'config': copy.deepcopy(config), 'selected': copy.deepcopy(selected), 'stats': stats,
                    'adapter': adapter.state_dict(), 'optimizer': optimizer.state_dict(),
                    'best_trial': saved, 'score_best': score_best, 'stale': stale, 'epoch_best': epoch_best,
                    'best': best, 'trials': trials, 'history': history, 'optimizer_steps': steps,
                    'excluded': failures, 'rng': grpo.rng_state(), 'decay_index': decay_index,
                    'epoch': epoch+1, 'trial_complete': done,
                    'complete': done and decay_index+1 == len(decays), 'boundary': 'epoch'})
            if done:
                break
    return finish(adapter)
