"""Shared checkpoint, RNG and signature helpers.

Discriminator token-level GRPO is implemented in tmm.discriminator. Finite-action
helpers below support earlier controller configurations.
"""
import hashlib
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def rng_state():
    """Capture training randomness without initializing CUDA for CPU learners."""
    state = {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
             'python': random.getstate()}
    if torch.cuda.is_initialized():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if state.get('cuda') is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def signature(value):
    """Hash configs and cached examples without expanding arrays into JSON."""
    checksum = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            checksum.update(str(('tensor', tensor.dtype, tuple(tensor.shape))).encode())
            checksum.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            checksum.update(str(('array', item.dtype, item.shape)).encode())
            checksum.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            checksum.update(b'dict')
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            checksum.update(str((type(item).__name__, len(item))).encode())
            for child in item:
                visit(child)
        else:
            checksum.update(str((type(item).__name__, item)).encode())
        checksum.update(b'\0')
    visit(value)
    return checksum.hexdigest()


def save_checkpoint(path, state):
    """Atomically replace a checkpoint only after a completed update boundary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.',
                                         suffix='.tmp', delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path, resume, schema, expected_signature):
    if resume and path is None:
        raise ValueError('resume requires checkpoint_path')
    if not resume or not Path(path).is_file():
        return None
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('schema') != schema or state.get('signature') != expected_signature:
        raise ValueError('Training checkpoint does not match this configuration and input data')
    return state


def validate(config):
    if int(config['group_size']) < 2:
        raise ValueError('GRPO group_size must be at least two')
    for key in ('temperature', 'max_grad_norm'):
        if float(config[key]) <= 0:
            raise ValueError(f'GRPO {key} must be positive')
    if not 0 < float(config['clip_epsilon']) < 1:
        raise ValueError('GRPO clip_epsilon must lie between zero and one')
    for key in ('weight', 'supervised_weight', 'kl_beta'):
        if float(config[key]) < 0:
            raise ValueError(f'GRPO {key} cannot be negative')
    if int(config['updates_per_group']) < 1:
        raise ValueError('GRPO updates_per_group must be positive')
    if config.get('max_steps') is not None and int(config['max_steps']) < 1:
        raise ValueError('GRPO max_steps must be positive or null')


def rollout(logits, rewards, reference_logits, group_size):
    """Sample with replacement from pi_old, preserving its detached log probs."""
    if logits.ndim != 1 or rewards.shape != logits.shape:
        raise ValueError('One reward is required per candidate, including STOP')
    if not torch.isfinite(logits).all() or not torch.isfinite(rewards).all():
        raise ValueError('GRPO logits and rewards must be finite')
    with torch.no_grad():
        old_log_probs = logits.log_softmax(0)
        actions = torch.multinomial(old_log_probs.exp(), int(group_size), replacement=True)
        sampled_rewards = rewards[actions]
        std = sampled_rewards.std(unbiased=False)
        advantages = (sampled_rewards - sampled_rewards.mean()) / (std + 1e-6)
        return {'actions': actions, 'old_log_probs': old_log_probs[actions],
                'advantages': advantages,
                'reference_log_probs': reference_logits.log_softmax(0).detach(),
                'reward_mean': float(sampled_rewards.mean()), 'reward_std': float(std)}


def objective(logits, sampled, clip_epsilon, kl_beta):
    """Clipped pi/pi_old surrogate plus exact KL(pi || pi_reference)."""
    log_probs = logits.log_softmax(0)
    ratio = (log_probs[sampled['actions']] - sampled['old_log_probs']).exp()
    advantages = sampled['advantages']
    direct = ratio * advantages
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages
    surrogate = -torch.minimum(direct, clipped).mean()
    kl = (log_probs.exp() * (log_probs - sampled['reference_log_probs'])).sum()
    loss = surrogate + kl_beta * kl
    return loss, {'grpo_loss': float(loss.detach()), 'kl': float(kl.detach()),
                  'clip_fraction': float(((ratio - 1).abs() > clip_epsilon).float().mean()),
                  'ratio_mean': float(ratio.detach().mean()),
                  'reward_mean': sampled['reward_mean'], 'reward_std': sampled['reward_std']}
