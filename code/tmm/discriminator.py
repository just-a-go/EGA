"""Screening token-policy GRPO on language-attention LoRA, followed by freezing.

The foundation, native perception/speech adapters, feature adapter and reader
stay fixed. Reference likelihoods disable only this module's new LoRA branch.
"""
import contextlib
import copy
import math
from pathlib import Path
import random
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .support import grpo
from .support.data import ObservationError, canonical, digest, pool, seed_views
from .support.screening import record_reward, schema_constraint, schema_token_choices


class LoRALinear(nn.Module):
    """Keep native linear/PEFT behavior and expose its original attributes."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        weight = base.weight
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=weight.device,
                                               dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=weight.device,
                                               dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale, self.enabled = float(alpha)/rank, True

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # These names belong to the new branch. Delegating them to a native
            # PEFT layer makes register_parameter() see its pre-existing LoRA.
            if name in ('base', 'lora_A', 'lora_B'):
                raise
            return getattr(super().__getattr__('base'), name)

    def forward(self, x, *args, **kwargs):
        original = self.base(x, *args, **kwargs)
        if not self.enabled:
            return original
        update = F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return original + update.to(original.dtype)*self.scale


def _replace(model, name, module):
    parent, _, child = name.rpartition('.')
    setattr(model.get_submodule(parent) if parent else model, child, module)


def reset_discriminator(backbone):
    for name, layer in getattr(backbone, '_ega_lora_modules', {}).items():
        _replace(backbone.model, name, layer.base)
    backbone._ega_lora_modules = {}
    backbone._ega_lora_scope = None
    backbone.discriminator_bundle = None
    backbone.discriminator_fingerprint = None
    backbone.discriminator_training = False
    backbone.model.eval().requires_grad_(False)


def _language_scope(backbone, configured=None):
    if configured:
        root = backbone.model.get_submodule(configured)
        if not hasattr(root, 'layers'):
            raise ValueError('language_scope must identify the language decoder containing layers')
        return configured
    for name in ('model.language_model', 'language_model', 'model.base_model.model', 'model',
                 'model.language_model.base_model.model'):
        try:
            root = backbone.model.get_submodule(name)
        except AttributeError:
            continue
        if hasattr(root, 'layers'):
            return name
    raise ValueError('Unsupported language decoder; specify its language_scope explicitly')


def _attach(backbone, config, module_names=None):
    if getattr(backbone, '_ega_lora_modules', None):
        reset_discriminator(backbone)
    scope = _language_scope(backbone, config.get('language_scope'))
    targets = set(config['target_modules'])
    pattern = re.compile(re.escape(scope)+r'\.layers\.\d+\.(?:self_attn|attn|attention)\.('+
                         '|'.join(re.escape(x) for x in sorted(targets))+r')$')
    names = [name for name, module in backbone.model.named_modules()
             if pattern.fullmatch(name) and hasattr(module, 'weight') and
             hasattr(module, 'in_features') and hasattr(module, 'out_features')]
    if not names or (module_names is not None and names != list(module_names)):
        raise ValueError('Language-only LoRA targets do not match this foundation checkpoint')
    modules = {}
    backbone.model.requires_grad_(False)
    for name in names:
        layer = LoRALinear(backbone.model.get_submodule(name), config['lora_rank'], config['lora_alpha'])
        _replace(backbone.model, name, layer)
        modules[name] = layer
    backbone._ega_lora_modules = modules
    backbone._ega_lora_scope = scope
    return names


def _state(backbone):
    return {name+'.'+key: getattr(layer, key).detach().cpu().clone()
            for name, layer in backbone._ega_lora_modules.items() for key in ('lora_A', 'lora_B')}


def _load_state(backbone, state):
    expected = {name+'.'+key for name in backbone._ega_lora_modules for key in ('lora_A', 'lora_B')}
    if set(state) != expected:
        raise ValueError('Discriminator LoRA state has missing or extra tensors')
    with torch.no_grad():
        for name, layer in backbone._ega_lora_modules.items():
            for key in ('lora_A', 'lora_B'):
                parameter = getattr(layer, key)
                value = state[name+'.'+key]
                if parameter.shape != value.shape:
                    raise ValueError('Discriminator LoRA tensor shape differs from this checkpoint')
                parameter.copy_(value.to(parameter.device, parameter.dtype))


@contextlib.contextmanager
def _reference(backbone):
    layers = list(backbone._ega_lora_modules.values())
    previous = [layer.enabled for layer in layers]
    for layer in layers:
        layer.enabled = False
    try:
        yield
    finally:
        for layer, enabled in zip(layers, previous):
            layer.enabled = enabled


def _train_parameters(backbone):
    # Native Phi set_adapter('speech') can re-enable its old adapter parameters.
    backbone.model.requires_grad_(False)
    parameters = []
    for layer in backbone._ega_lora_modules.values():
        layer.lora_A.requires_grad_(True)
        layer.lora_B.requires_grad_(True)
        parameters.extend((layer.lora_A, layer.lora_B))
    return parameters


def _reset_rope(backbone):
    if hasattr(backbone.model, 'rope_deltas'):
        backbone.model.rope_deltas = None


def completion_log_probs(backbone, batch, tokens, temperature, constrained_json=False):
    """Teacher-force only completion likelihoods, retaining native media inputs."""
    tokens = tokens.reshape(1, -1).to(backbone.device)
    length = tokens.shape[1]
    if not length:
        raise ValueError('A generated completion must include at least its EOS token')
    inputs = dict(batch)
    if 'input_ids' in inputs:
        inputs['input_ids'] = torch.cat([inputs['input_ids'], tokens], dim=1)
    elif 'inputs_embeds' in inputs:
        generated = backbone.model.get_input_embeddings()(tokens)
        inputs['inputs_embeds'] = torch.cat([inputs['inputs_embeds'].detach(), generated], dim=1)
    else:
        raise ValueError('Native discriminator batch exposes neither input_ids nor inputs_embeds')
    if 'attention_mask' in inputs:
        mask = inputs['attention_mask']
        inputs['attention_mask'] = torch.cat([mask, mask.new_ones((1, length))], dim=-1)
    if 'position_ids' in inputs:
        ids = inputs['position_ids']
        inputs['position_ids'] = torch.cat([ids, ids[..., -1:] +
                                            torch.arange(1, length+1, device=ids.device)], dim=-1)
    if 'token_type_ids' in inputs:
        types = inputs['token_type_ids']
        inputs['token_type_ids'] = torch.cat([types, types.new_zeros((1, length))], dim=-1)
    if getattr(backbone, 'phi_remote', False):
        inputs['num_logits_to_keep'] = length+1
    _reset_rope(backbone)
    logits = backbone.model(**inputs, use_cache=False).logits
    if logits.shape[1] < length+1:
        raise ValueError('Native model does not expose all teacher-forced completion logits')
    logits = logits[0, -length-1:-1].float()/temperature
    if constrained_json:
        choices = schema_token_choices(backbone.tokenizer, backbone.screening, backbone.label_ids)
        constrained = torch.full_like(logits, -torch.inf)
        for i in range(length):
            allowed = choices[i] if i < len(choices) else [backbone.tokenizer.eos_token_id]
            constrained[i, allowed] = logits[i, allowed]
        logits = constrained
    values = logits.log_softmax(-1).gather(1, tokens[0, :, None]).squeeze(1)
    if not torch.isfinite(values).all():
        raise RuntimeError('Non-finite generated-token likelihood')
    return values


def token_grpo_loss(log_probs, old_log_probs, reference_log_probs, mask, advantage, clip_epsilon, kl_beta):
    """Clipped token ratio plus sampled k3 KL, normalized by response length."""
    if mask.shape != log_probs.shape or not bool(mask.sum()):
        raise ValueError('GRPO needs a nonempty completion-token mask')
    ratio = (log_probs-old_log_probs).exp()
    surrogate = torch.minimum(ratio*advantage, ratio.clamp(1-clip_epsilon, 1+clip_epsilon)*advantage)
    difference = reference_log_probs-log_probs
    kl = difference.exp()-difference-1
    loss = ((-surrogate+kl_beta*kl)*mask).sum()/mask.sum()
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite discriminator GRPO objective')
    return loss, {'kl': float((kl.detach()*mask).sum()/mask.sum()),
                  'clip_fraction': float((((ratio.detach()-1).abs()>clip_epsilon)*mask).sum()/mask.sum())}


def _sample(backbone, batch, config):
    from transformers import StoppingCriteria, StoppingCriteriaList
    prefix = batch['input_ids'].shape[1] if 'input_ids' in batch else 0
    class CompleteJson(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            import json
            answer = backbone.tokenizer.decode(input_ids[0, prefix:], skip_special_tokens=True).strip()
            try:
                return isinstance(json.loads(answer), dict)
            except ValueError:
                return False
    options = {'stopping_criteria': StoppingCriteriaList([CompleteJson()])}
    if config['constrained_json']:
        options['prefix_allowed_tokens_fn'] = schema_constraint(backbone.tokenizer, backbone.screening,
                                                                backbone.label_ids, prefix)
    if getattr(backbone, 'phi_remote', False):
        options['num_logits_to_keep'] = 1
    _reset_rope(backbone)
    backbone.model.eval()
    with torch.no_grad():
        output = backbone.model.generate(**batch, do_sample=True, temperature=config['temperature'],
            top_p=1., top_k=0, max_new_tokens=config['max_new_tokens'],
            repetition_penalty=config['repetition_penalty'], no_repeat_ngram_size=0,
            forced_bos_token_id=None, forced_eos_token_id=None,
            return_dict_in_generate=True, output_scores=True, **options)
    count = len(output.scores)
    if not count:
        raise RuntimeError('Native generation returned no completion tokens')
    tokens = output.sequences[0, -count:].detach()
    eos = getattr(backbone.model.generation_config, 'eos_token_id', backbone.tokenizer.eos_token_id)
    eos = eos if isinstance(eos, (list, tuple)) else [eos]
    mask = torch.ones(count, dtype=torch.float32, device=tokens.device)
    endings = [i for i, token in enumerate(tokens.tolist()) if token in eos]
    if endings:
        mask[endings[0]+1:] = 0
    answer = backbone.tokenizer.decode(tokens[mask.bool()], skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
    return tokens, mask, answer


def _stable_records(records):
    return [{k: v for k, v in rec.items() if k not in ('index_seconds', 'views')} |
            {'views': [{k: v for k, v in view.items() if k != 'reader_positions'} for view in rec['views']]}
            for rec in records]


def _views(record, track, length=0):
    initial = seed_views(record, track)
    options = [view for view in pool(record, track) if view not in initial]
    return canonical(initial+options[:length])


def _prompt(backbone, views, mode='reader'):
    if mode not in ('plain', 'reader'):
        raise ValueError('Discriminator prompt mode must be plain or reader')
    evidence = backbone.reader.read(views) if mode == 'reader' else None
    method = getattr(backbone, 'discriminator_batch', None)
    if method is None:
        raise ValueError('Backbone must expose a differentiable discriminator_batch interface')
    batch, _ = method(views, evidence=evidence, use_reader=mode == 'reader')
    return {k: v.detach() if torch.is_tensor(v) else v for k, v in batch.items()}


def _prompt_mode(config, group_index, epoch):
    # Absolute group indices make this schedule stable across interrupted chunks.
    # Each epoch rotates modes without adding prompts or optimizer updates.
    modes = config['prompt_modes']
    return modes[(group_index+epoch) % len(modes)]


def _validation_reward(backbone, records, track, modes, excluded):
    values = {}
    backbone.model.eval()
    for rec in records:
        record_values = {}
        for mode in modes:
            try:
                prediction = backbone.score(_views(rec, track), use_reader=mode == 'reader')
            except (ObservationError, FileNotFoundError) as exc:
                excluded.append({'stage': 'dev', 'record': rec['id'], 'prompt_mode': mode, 'error': str(exc)})
                continue
            if not prediction.get('output_generated', False):
                excluded.append({'stage': 'dev', 'record': rec['id'], 'prompt_mode': mode,
                                 'error': prediction.get('error')})
                continue
            record_values[mode] = record_reward(prediction['answer'], rec, backbone.screening)['total']
        # Compare modes on the same available source records; a failed input
        # must not silently change their weights in checkpoint selection.
        if len(record_values) == len(modes):
            participant = values.setdefault((rec['family'], rec['participant']), {mode: [] for mode in modes})
            for mode, value in record_values.items():
                participant[mode].append(value)
    if not values:
        raise ValueError('No paired generated development outputs for discriminator checkpoint selection')
    mode_rewards = {mode: float(np.mean([np.mean(v[mode]) for v in values.values()])) for mode in modes}
    return float(np.mean(list(mode_rewards.values()))), mode_rewards


def _policy_mode(backbone, checkpointing):
    # Train mode enables HF non-reentrant activation checkpointing. Dropout is
    # disabled so sampling, behavior, current and reference likelihoods agree.
    backbone.model.eval()
    decoder = backbone.model.get_submodule(backbone._ega_lora_scope)
    # Enable checkpointing in the language decoder without training perception
    # modules nested inside native embedding implementations (notably Phi).
    decoder.training = bool(checkpointing)
    for layer in decoder.layers:
        layer.train(checkpointing)
    for module in decoder.layers.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.
        for name in ('attention_dropout', 'dropout'):
            if isinstance(getattr(module, name, None), float):
                setattr(module, name, 0.)


def _validate(config):
    config = dict(config)
    defaults = {'enabled': True, 'algorithm': 'token_grpo_lora', 'learning_rate': 1e-5,
        'epochs': 3, 'patience': 1, 'group_size': 4, 'temperature': .8, 'top_p': 1., 'top_k': 0,
        'clip_epsilon': .2, 'kl_beta': .04, 'max_new_tokens': 128,
        'gradient_accumulation_groups': 4, 'updates_per_group': 2, 'max_grad_norm': 1.,
        'gradient_checkpointing': True, 'lora_rank': 16, 'lora_alpha': 32,
        'lora_dropout': 0., 'max_steps': None, 'prefix_lengths': [0, 1, 3],
        'constrained_json': False, 'weight_decay': 0., 'repetition_penalty': 1.,
        'prompt_modes': ['plain', 'reader']}
    for key, value in defaults.items():
        config.setdefault(key, value)
    if not config['enabled'] or config['algorithm'] != 'token_grpo_lora':
        raise ValueError('The Screening discriminator phase requires token_grpo_lora')
    if (config['top_p'] != 1 or config['top_k'] != 0 or config['lora_dropout'] != 0
            or config['repetition_penalty'] != 1):
        raise ValueError('Use top_p=1, top_k=0, repetition_penalty=1 and LoRA dropout=0 for matching likelihoods')
    for key in ('learning_rate', 'temperature', 'max_grad_norm', 'lora_alpha'):
        if not math.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f'Discriminator {key} must be positive and finite')
    for key in ('epochs', 'patience', 'max_new_tokens', 'gradient_accumulation_groups', 'updates_per_group', 'lora_rank'):
        if int(config[key]) < 1:
            raise ValueError(f'Discriminator {key} must be positive')
    if int(config['group_size']) < 2 or not 0 < config['clip_epsilon'] < 1 or config['kl_beta'] < 0:
        raise ValueError('Invalid discriminator GRPO settings')
    if config['max_steps'] is not None and int(config['max_steps']) < 1:
        raise ValueError('Discriminator max_steps must be positive or None')
    if not config.get('target_modules') or set(config['target_modules'])-{'q_proj','v_proj','k_proj','o_proj','qkv_proj'}:
        raise ValueError('Only explicit language-attention projection targets are supported')
    if not config['prefix_lengths'] or any(type(n) is not int or n < 0 for n in config['prefix_lengths']):
        raise ValueError('Discriminator prefixes must be nonnegative integers')
    modes = config['prompt_modes']
    if (not isinstance(modes, (list, tuple)) or not modes or
            any(mode not in ('plain', 'reader') for mode in modes) or len(set(modes)) != len(modes)):
        raise ValueError('Discriminator prompt_modes must list unique plain/reader modes')
    config['prompt_modes'] = list(modes)
    return config


def _fingerprint(bundle):
    return grpo.signature((bundle['schema'], bundle['backbone_spec'], bundle['config'],
                           bundle['module_names'], bundle['state_dict']))


def restore_discriminator(backbone, bundle):
    if bundle.get('schema') != 'discriminator-lora-v1' or bundle['backbone_spec'] != backbone.spec:
        raise ValueError('Discriminator bundle requires its original foundation checkpoint')
    if bundle['fingerprint'] != _fingerprint(bundle):
        raise ValueError('Discriminator bundle fingerprint differs from its weights/config')
    _attach(backbone, bundle['config'], bundle['module_names'])
    _load_state(backbone, bundle['state_dict'])
    backbone.model.eval().requires_grad_(False)
    backbone.discriminator_training = False
    backbone.discriminator_bundle = bundle
    backbone.discriminator_fingerprint = bundle['fingerprint']


def train_discriminator(backbone, train, dev, track, config, checkpoint_dir=None,
                        resume=False, seed=27, selected=None):
    config = _validate(config)
    if getattr(backbone, 'screening', None) is None or getattr(backbone, 'reader', None) is None:
        raise ValueError('Discriminator GRPO requires the Screening schema and a trained frozen Reader')
    if getattr(backbone, 'kind', None) == 'minicpm':
        raise ValueError('MiniCPM custom chat has no supported differentiable discriminator batch')
    epochs = int(selected['epochs'] if selected else config['epochs'])
    if epochs < 1:
        raise ValueError('Selected discriminator epochs must be positive')
    groups = {}
    for rec in train:
        if rec.get('split') in ('test', 'unused'):
            raise ValueError('Test/audit participants cannot train the discriminator')
        groups.setdefault((rec['family'], rec['participant']), []).append(rec)
    if not groups or (not dev and selected is None):
        raise ValueError('Discriminator GRPO needs training and development participants')
    if selected is None and set(groups) & {(r['family'], r['participant']) for r in dev}:
        raise ValueError('Discriminator train/development participants overlap')
    reader = backbone.reader
    reader.model.eval().requires_grad_(False)
    if getattr(backbone, 'adapter', None) is not None:
        backbone.adapter.eval().requires_grad_(False)
    expected = grpo.signature((config, epochs, seed, track, backbone.spec,
                               getattr(reader, 'fingerprint', None), _stable_records(train), _stable_records(dev)))
    checkpoint = Path(checkpoint_dir)/'training.pt' if checkpoint_dir is not None else None
    restored = grpo.load_checkpoint(checkpoint, resume, 'discriminator-training-v1', expected)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    names = _attach(backbone, config)
    initial = _state(backbone)
    parameters = _train_parameters(backbone)
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    if config['gradient_checkpointing']:
        backbone.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    backbone.discriminator_training = True
    step, history, excluded, start_epoch, next_chunk = 0, [], [], 0, 0
    best_score, best_epoch, best, stale, complete = -float('inf'), 0, None, 0, False
    best_mode_rewards = {}
    if restored:
        _load_state(backbone, restored['lora'])
        optimizer.load_state_dict(restored['optimizer'])
        initial, step, history, excluded = (restored[k] for k in ('initial', 'step', 'history', 'excluded'))
        start_epoch, next_chunk = restored['epoch'], restored['next_chunk']
        best_score, best_epoch, best, stale, complete = (restored[k] for k in ('best_score','best_epoch','best','stale','complete'))
        best_mode_rewards = restored['best_mode_rewards']
        grpo.restore_rng(restored['rng'])

    def save(epoch, chunk, done):
        if checkpoint is not None:
            grpo.save_checkpoint(checkpoint, {'schema': 'discriminator-training-v1', 'signature': expected,
                'lora': _state(backbone), 'optimizer': optimizer.state_dict(), 'initial': initial,
                'epoch': epoch, 'next_chunk': chunk, 'step': step, 'history': history,
                'excluded': excluded, 'best_score': best_score, 'best_epoch': best_epoch,
                'best': best, 'best_mode_rewards': best_mode_rewards,
                'stale': stale, 'complete': done, 'rng': grpo.rng_state()})

    participants = list(groups.values())
    chunks = [participants[i:i+config['gradient_accumulation_groups']]
              for i in range(0, len(participants), config['gradient_accumulation_groups'])]
    if restored is None:
        save(0, 0, False)
    try:
        for epoch in range(start_epoch, epochs) if not complete else ():
            for chunk_index in range(next_chunk if epoch == start_epoch else 0, len(chunks)):
                if config['max_steps'] is not None and step >= config['max_steps']:
                    break
                sampled = []
                for group_offset, records in enumerate(chunks[chunk_index]):
                    rec = random.choice(records)
                    mode = _prompt_mode(config, chunk_index*config['gradient_accumulation_groups']+group_offset, epoch)
                    try:
                        views = _views(rec, track, random.choice(config['prefix_lengths']))
                        batch = _prompt(backbone, views, mode)
                    except (ObservationError, FileNotFoundError) as exc:
                        excluded.append({'stage': 'train', 'record': rec['id'], 'prompt_mode': mode, 'error': str(exc)})
                        continue
                    rollouts = []
                    for _ in range(config['group_size']):
                        tokens, mask, answer = _sample(backbone, batch, config)
                        _policy_mode(backbone, config['gradient_checkpointing'])
                        with torch.no_grad():
                            old = completion_log_probs(backbone, batch, tokens, config['temperature'], config['constrained_json'])
                            with _reference(backbone):
                                reference = completion_log_probs(backbone, batch, tokens, config['temperature'], config['constrained_json'])
                        reward = record_reward(answer, rec, backbone.screening)
                        rollouts.append({'tokens': tokens, 'mask': mask, 'old': old, 'reference': reference,
                                         'answer': answer, 'reward': reward, 'prompt_mode': mode})
                    rewards = torch.tensor([r['reward']['total'] for r in rollouts], device=backbone.device)
                    advantages = (torch.zeros_like(rewards) if bool(rewards.max() == rewards.min()) else
                                  (rewards-rewards.mean())/(rewards.std(unbiased=False)+1e-6))
                    sampled.append((batch, rollouts, advantages))
                if not sampled:
                    continue
                for update in range(config['updates_per_group']):
                    optimizer.zero_grad()
                    _train_parameters(backbone)
                    _policy_mode(backbone, config['gradient_checkpointing'])
                    statistics, losses = [], []
                    for batch, rollouts, advantages in sampled:
                        for rollout, advantage in zip(rollouts, advantages):
                            current = completion_log_probs(backbone, batch, rollout['tokens'], config['temperature'], config['constrained_json'])
                            loss, detail = token_grpo_loss(current, rollout['old'], rollout['reference'], rollout['mask'],
                                                          advantage, config['clip_epsilon'], config['kl_beta'])
                            (loss/(len(sampled)*config['group_size'])).backward()
                            statistics.append(detail)
                            losses.append(float(loss.detach()))
                    norm = torch.nn.utils.clip_grad_norm_(parameters, config['max_grad_norm'], error_if_nonfinite=True)
                    optimizer.step()
                    if not all(torch.isfinite(p).all() for p in parameters):
                        raise RuntimeError('Non-finite discriminator LoRA update')
                    step += 1
                    row = {'step': step, 'epoch': epoch+1, 'chunk': chunk_index, 'loss': float(np.mean(losses)),
                           'gradient_norm': float(norm), 'kl': float(np.mean([d['kl'] for d in statistics])),
                           'clip_fraction': float(np.mean([d['clip_fraction'] for d in statistics])),
                           'reward_mean': float(np.mean([r['reward']['total'] for _, rs, _ in sampled for r in rs])),
                           'reward_std': float(np.mean([np.std([r['reward']['total'] for r in rs]) for _, rs, _ in sampled])),
                            'prompt_groups': {mode: sum(rs[0]['prompt_mode'] == mode for _, rs, _ in sampled)
                                              for mode in config['prompt_modes']},
                            'raw_rollouts': [{'answer': r['answer'], 'reward': r['reward'], 'prompt_mode': r['prompt_mode']}
                                             for _, rs, _ in sampled for r in rs]}
                    history.append(row)
                    print('discriminator_update: '+str({k: v for k, v in row.items() if k != 'raw_rollouts'}), flush=True)
                    if config['max_steps'] is not None and step >= config['max_steps']:
                        break
                save(epoch, chunk_index+1, False)
                if config['max_steps'] is not None and step >= config['max_steps']:
                    break
            if step == 0:
                raise ValueError('No usable discriminator training groups completed an update')
            value, mode_rewards = _validation_reward(backbone, dev or train, track, config['prompt_modes'], excluded)
            if selected is not None or value > best_score:
                best_score, best_epoch, best, stale = value, epoch+1, _state(backbone), 0
                best_mode_rewards = mode_rewards
            else:
                stale += 1
            complete = (epoch+1 == epochs or (selected is None and stale >= config['patience']) or
                        (config['max_steps'] is not None and step >= config['max_steps']))
            next_chunk = 0
            save(epoch+1, 0, complete)
            if complete:
                break
        if best is None:
            raise ValueError('Discriminator has no completed selection epoch')
        delta = math.sqrt(sum(float((best[k]-initial[k]).square().sum()) for k in best))
        bundle = {'schema': 'discriminator-lora-v1', 'adapter_name': 'ega_discriminator', 'config': config,
                  'backbone_spec': copy.deepcopy(backbone.spec), 'module_names': names, 'state_dict': best,
                  'epochs': best_epoch, 'dev_reward': best_score, 'history': history, 'optimizer_steps': step,
                  'dev_mode_rewards': best_mode_rewards,
                  'parameter_delta_norm': delta, 'excluded': excluded, 'seed': seed,
                  'reader_fingerprint': getattr(reader, 'fingerprint', None), 'resumed': restored is not None,
                  'algorithm': 'token_grpo_lora'}
        bundle['fingerprint'] = _fingerprint(bundle)
        _load_state(backbone, best)
        backbone.discriminator_bundle, backbone.discriminator_fingerprint = bundle, bundle['fingerprint']
        if checkpoint_dir is not None:
            grpo.save_checkpoint(Path(checkpoint_dir)/'adapter.pt', bundle)
        return bundle
    finally:
        backbone.discriminator_training = False
        backbone.model.eval().requires_grad_(False)
        if config['gradient_checkpointing']:
            backbone.model.gradient_checkpointing_disable()


def inspect_selection_candidates(training, frontend, train, dev, track, seed):
    """Validate completed source training and expose only best/current adapters."""
    discriminator, reader = frontend['discriminator'], frontend['reader']
    if (frontend.get('schema') != 'ega-v2' or discriminator.get('schema') != 'discriminator-lora-v1'
            or training.get('schema') != 'discriminator-training-v1'):
        raise ValueError('Source reselection requires the original complete Screening/Reader checkpoints')
    if (not training.get('complete') or training.get('next_chunk') != 0
            or not 1 <= training.get('best_epoch', 0) <= training.get('epoch', 0)
            or training.get('step', 0) <= 0):
        raise ValueError('Only completed epoch-boundary best/current checkpoints may be reselected')
    config = _validate(discriminator['config'])
    if (discriminator['seed'] != seed or reader['seed'] != seed
            or discriminator['reader_fingerprint'] != reader['fingerprint']
            or discriminator['backbone_spec'] != reader['backbone_spec']
            or discriminator['fingerprint'] != _fingerprint(discriminator)):
        raise ValueError('Discriminator foundation, Reader, seed or saved fingerprint changed')
    from .reader import reader_fingerprint
    if reader['fingerprint'] != reader_fingerprint(reader):
        raise ValueError('Original Reader checkpoint fingerprint changed')
    for split, records in (('train', train), ('dev', dev)):
        if not records or any(r.get('split') != split or r['label'] not in (0, 1) for r in records):
            raise ValueError('Reselection uses only the original source train/development partitions')
        roster = sorted({r['family']+':'+r['participant'] for r in records})
        if (frontend[split+'_participants'] != roster
                or sorted(':'.join(p) for p in reader[split+'_participants']) != roster):
            raise ValueError('Source/Reader participant partitions changed')
    if {(r['family'], r['participant']) for r in train} & {(r['family'], r['participant']) for r in dev}:
        raise ValueError('Source train and development participants overlap')
    expected = grpo.signature((config, config['epochs'], seed, track, discriminator['backbone_spec'],
                               reader['fingerprint'], _stable_records(train), _stable_records(dev)))
    if training['signature'] != expected:
        raise ValueError('Original discriminator training signature or source observations changed')
    if (training['epoch'] > config['epochs'] or training['best_epoch'] != discriminator['epochs']
            or grpo.signature(training['best']) != grpo.signature(discriminator['state_dict'])):
        raise ValueError('Completed training best does not match the exported frontend adapter')
    candidates, seen = [], {}
    for name, epoch, state in (('best', training['best_epoch'], training['best']),
                               ('current', training['epoch'], training['lora'])):
        if set(state) != set(training['initial']):
            raise ValueError('Candidate language-LoRA tensor identities changed')
        for key, value in state.items():
            if value.shape != training['initial'][key].shape or not torch.isfinite(value).all():
                raise ValueError('Candidate LoRA contains incompatible or nonfinite weights')
        identity = grpo.signature(state)
        candidates.append({'name': name, 'epoch': epoch, 'weights_sha256': identity,
                           'duplicate_of': seen.get(identity), 'state_dict': state})
        seen.setdefault(identity, name)
    return candidates, {'training_signature': expected, 'epoch': training['epoch'],
                        'best_epoch': training['best_epoch'], 'step': training['step'],
                        'next_chunk': training['next_chunk'], 'complete': training['complete'],
                        'reader_fingerprint': reader['fingerprint'],
                        'source_signature': grpo.signature((_stable_records(train), _stable_records(dev)))}


def selection_views(record, prefix):
    """Fixed seed plus a source-hash prefix of same-event native audio."""
    from .controller import aligned_audio_candidates
    initial = canonical(seed_views(record, 'A+T'))
    if len(initial) != 3 or any(v['modality'] != 'T' for v in initial):
        raise ValueError('This source selection requires its original three text seeds')
    remaining = [v for v in pool(record, 'A+T') if v['modality'] == 'A' and v not in initial]
    selected = aligned_audio_candidates(record, initial, remaining)[:prefix]
    if len(selected) != prefix:
        raise ValueError('Source development does not cover the requested same-event audio prefix')
    return canonical(initial+selected)


def select_source_candidates(backbone, records, candidates, discriminator, selection, directory,
                             resume=False, seed_cache=None, context_identity=None):
    """Evaluate the fixed six source cells once; no optimizer or cutoff fitting."""
    import hashlib
    import json
    if (selection.get('prefix_lengths') != [0, 1, 3]
            or selection.get('prompt_modes') != ['plain', 'reader']
            or selection.get('metric') != 'class_balanced_raw_brier'
            or selection.get('audio_alignment') != 'same_event_seed'
            or selection.get('audio_order') != 'source_hash'
            or selection.get('candidates') != ['best', 'current']):
        raise ValueError('Use the declared fixed best/current six-cell source selection')
    if not records or any(r.get('split') != 'dev' for r in records):
        raise ValueError('Candidate selection is restricted to the original source development records')
    directory = Path(directory)
    reports, files = [], {}
    for candidate in candidates:
        if candidate['duplicate_of'] is not None:
            continue
        adapted = dict(discriminator, state_dict=candidate['state_dict'], epochs=candidate['epoch'])
        adapted['fingerprint'] = _fingerprint(adapted)
        restore_discriminator(backbone, adapted)
        cells = {(prefix, mode): {} for prefix in selection['prefix_lengths'] for mode in selection['prompt_modes']}
        invalid = []
        for number, rec in enumerate(records, 1):
            key = digest((rec['family'], rec['id']))[:20]
            path = directory/candidate['weights_sha256']/(key+'.pt')
            expected = grpo.signature((candidate['weights_sha256'], adapted['fingerprint'],
                                       backbone.reader.fingerprint, backbone.protocol, backbone.instruction,
                                       selection, context_identity, _stable_records([rec])))
            cached = grpo.load_checkpoint(path, resume, 'discriminator-source-selection-record-v1', expected)
            cached = cached or {'schema': 'discriminator-source-selection-record-v1', 'signature': expected, 'cells': {}}
            for prefix in selection['prefix_lengths']:
                views = selection_views(rec, prefix)
                for mode in selection['prompt_modes']:
                    cell = f'{prefix}:{mode}'
                    reused = cell in cached['cells']
                    if not reused:
                        legacy = (seed_cache or {}).get(mode, {}) if candidate['name'] == 'best' and prefix == 0 else {}
                        prediction = legacy.get('predictions', {}).get((rec['family'], rec['id']))
                        source = legacy.get('source') if prediction is not None else None
                        if prediction is None:
                            evidence = backbone.reader.read(views) if mode == 'reader' else None
                            if mode == 'reader' and not evidence:
                                prediction = cached['cells'][f'{prefix}:plain']['prediction']
                                source = 'identical_plain_empty_reader'
                            else:
                                prediction = backbone.score(views, evidence=evidence, use_reader=mode == 'reader')
                        cached['cells'][cell] = {'prediction': prediction, 'reused_from': source,
                                                 'view_ids': [v['id'] for v in views]}
                        grpo.save_checkpoint(path, cached)
                    prediction = cached['cells'][cell]['prediction']
                    valid = (prediction.get('status') == 'ok' and prediction.get('probability_valid') is True
                             and isinstance(prediction.get('p'), (float, int)) and 0 <= prediction['p'] <= 1)
                    if not valid:
                        invalid.append({'record': rec['id'], 'prefix': prefix, 'mode': mode,
                                        'status': prediction.get('status'), 'error': prediction.get('error')})
                    else:
                        person = (rec['family'], rec['participant'])
                        values = cells[(prefix, mode)].setdefault(person, {'label': rec['label'], 'losses': [], 'rewards': []})
                        if values['label'] != rec['label']:
                            raise ValueError('Source participant labels changed between records')
                        values['losses'].append((prediction['p']-rec['label'])**2)
                        values['rewards'].append(record_reward(prediction['answer'], rec, backbone.screening)['total'])
                    print('discriminator_selection_progress: '+json.dumps({
                        'candidate': candidate['name'], 'prefix': prefix, 'mode': mode,
                        'record': rec['id'], 'done': number, 'total': len(records),
                        'cached_or_reused': reused or bool(cached['cells'][cell]['reused_from'])}), flush=True)
            files[str(path.relative_to(directory)).replace('\\', '/')] = hashlib.sha256(path.read_bytes()).hexdigest()
        summaries = []
        if not invalid:
            for (prefix, mode), people in cells.items():
                classes = [[float(np.mean(v['losses'])) for v in people.values() if v['label'] == label] for label in (0, 1)]
                if not all(classes):
                    raise ValueError('Class-balanced selection requires both native source classes in every cell')
                summaries.append({'prefix': prefix, 'mode': mode, 'participants': len(people),
                                  'class_counts': [len(group) for group in classes],
                                  'balanced_brier': .5*(float(np.mean(classes[0]))+float(np.mean(classes[1]))),
                                  'json_reward': float(np.mean([np.mean(v['rewards']) for v in people.values()]))})
        reports.append({'candidate': candidate['name'], 'epoch': candidate['epoch'],
                        'weights_sha256': candidate['weights_sha256'], 'valid': not invalid,
                        'mean_balanced_brier': float(np.mean([v['balanced_brier'] for v in summaries])) if summaries else None,
                        'cells': summaries, 'invalid': invalid})
    valid_reports = [report for report in reports if report['valid']]
    if not valid_reports:
        raise ValueError('Neither completed candidate has valid full six-cell source coverage')
    winner = min(valid_reports, key=lambda r: (r['mean_balanced_brier'], r['candidate'] != 'best'))
    return winner, reports, files
