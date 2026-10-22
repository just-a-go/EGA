"""One fixed, single-label-token GRPO adaptation for EMER independent QA.

The action space is exactly the native 0/1 label tokens. Sampling and old,
reference and current likelihoods all use the same two-logit softmax at T=1.
Only the supplied TRAIN labels enter rewards; the Reader remains frozen.
"""
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from ..discriminator import (_attach, _state, _load_state, _reference,
                             _train_parameters, _policy_mode, token_grpo_loss)
from . import grpo


ALGORITHM = "token_grpo_lora_binary_qa_v1"
DEFAULTS = {"seed": 27, "group_size": 32, "max_updates": 24, "max_groups": 96,
            "clip_epsilon": .2, "kl_beta": .01, "learning_rate": 1e-5,
            "lora_rank": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"],
            "gradient_checkpointing": True, "max_grad_norm": 1., "temperature": 1.}
MODES = ((), ("A",), ("V",), ("A", "V"))


def _pair_log_probs(backbone, batch):
    if hasattr(backbone.model, "rope_deltas"):
        backbone.model.rope_deltas = None
    # Retain every prompt state for attention/checkpointing, while avoiding an
    # unused sequence-length x vocabulary projection in old/ref/current passes.
    # Slicing the head input preserves both the final logits and their gradients.
    def final_position(module, inputs):
        return (inputs[0][:, -1:, :], *inputs[1:])
    handle = backbone.model.lm_head.register_forward_pre_hook(final_position)
    try:
        output = backbone.model(**batch, use_cache=False)
    finally:
        handle.remove()
    pair = output.logits[0, -1, backbone.label_ids].float()
    if pair.shape != (2,) or not bool(torch.isfinite(pair).all()):
        raise RuntimeError("GRPO needs finite logits for exactly the native 0/1 tokens")
    return pair.log_softmax(-1)


def _observed_evidence(qa, item, acquired, evidence_by_id):
    value = evidence_by_id.get(item["id"], [])
    if isinstance(value, dict):
        value = value.get("T"+"".join(acquired), [])
    if not isinstance(value, list):
        raise ValueError("TRAIN Reader evidence must be a list or a mode-keyed record dictionary")
    observed = {v["id"] for v in qa.observed_views(item, acquired)}
    return [record for record in value if record.get("source_id") in observed]


def train(qa, train_items, labels_by_id, evidence_by_id, outdir, config=None):
    """Train/freeze 24 real LoRA updates, skipping zero-variance groups.

    ``train_items`` contains only EmerQA model-visible dictionaries. Labels are
    separate six-element 0/1 arrays. Existing exact-signature checkpoints are
    resumed at completed optimizer boundaries; no checkpoint is silently reset.
    """
    cfg = DEFAULTS | (config or {})
    if (float(cfg["temperature"]) != 1. or int(cfg["group_size"]) < 2
            or not 0 < float(cfg["clip_epsilon"]) < 1 or float(cfg["kl_beta"]) < 0
            or int(cfg["max_updates"]) < 1 or int(cfg["max_groups"]) < 1):
        raise ValueError("Invalid fixed EMER binary token GRPO settings")
    items = list(train_items)
    ids = [item["id"] for item in items]
    if not items or len(set(ids)) != len(ids) or set(labels_by_id) != set(ids):
        raise ValueError("GRPO requires unique TRAIN items and exactly their separate label arrays")
    for item in items:
        qa._validate(item)
        labels = np.asarray(labels_by_id[item["id"]])
        if labels.shape != (6,) or not np.isin(labels, (0, 1)).all():
            raise ValueError("GRPO TRAIN labels must be six independent binary values")
    backbone = qa.backbone
    for value, token_id in zip(("0", "1"), backbone.label_ids):
        if backbone.tokenizer.encode(value, add_special_tokens=False) != [int(token_id)]:
            raise ValueError("Native binary label token IDs do not match the inference scorer")
    reader = getattr(backbone, "reader", None)
    if reader is not None and hasattr(reader, "model"):
        reader.model.eval().requires_grad_(False)
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path, checkpoint_path = output_dir/"discriminator.pt", output_dir/"training.pt"
    expected = grpo.signature((ALGORITHM, cfg, getattr(backbone, "spec", None), items,
                               labels_by_id, evidence_by_id, getattr(reader, "fingerprint", None)))
    random.seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    names = _attach(backbone, cfg)
    initial = _state(backbone)
    if final_path.is_file():
        saved = torch.load(final_path, map_location="cpu", weights_only=False)
        if saved.get("signature") != expected or saved.get("algorithm") != ALGORITHM:
            raise ValueError("Existing EMER GRPO discriminator differs from the current inputs/settings")
        _load_state(backbone, saved["state_dict"])
        backbone.model.eval().requires_grad_(False)
        backbone.discriminator_fingerprint = saved["fingerprint"]
        backbone.discriminator_bundle = saved
        backbone.discriminator_training = False
        return dict(saved["report"], resumed_completed=True)
    parameters = _train_parameters(backbone)
    optimizer = torch.optim.Adam(parameters, lr=float(cfg["learning_rate"]))
    if cfg["gradient_checkpointing"]:
        backbone.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone.discriminator_training = True
    updates, next_group, history = 0, 0, []
    restored = grpo.load_checkpoint(checkpoint_path, True, "emer-grpo-training-v1", expected)
    if restored:
        _load_state(backbone, restored["state_dict"])
        optimizer.load_state_dict(restored["optimizer"])
        initial = restored["initial"]
        updates, next_group, history = restored["updates"], restored["next_group"], restored["history"]
        grpo.restore_rng(restored["rng"])
    # Shuffle only TRAIN source/option IDs, independent of labels and outcomes.
    ordered = sorted([(item, option) for item in items for option in range(6)],
                     key=lambda row: (row[0]["id"], row[1]))
    random.Random(int(cfg["seed"])).shuffle(ordered)

    def save(group_index):
        grpo.save_checkpoint(checkpoint_path, {"schema": "emer-grpo-training-v1", "signature": expected,
            "state_dict": _state(backbone), "optimizer": optimizer.state_dict(), "initial": initial,
            "updates": updates, "next_group": group_index, "history": history, "rng": grpo.rng_state()})

    try:
        for group_index in range(next_group, int(cfg["max_groups"])):
            if updates >= int(cfg["max_updates"]):
                break
            item, option = ordered[group_index % len(ordered)]
            acquired = MODES[group_index % len(MODES)]
            if ("A" in acquired and item.get("audio") is None
                    or "V" in acquired and not item.get("frames")):
                raise ValueError("The fixed T/TA/TV/TAV GRPO schedule requires audio and frames for every TRAIN item")
            evidence = _observed_evidence(qa, item, acquired, evidence_by_id)
            batch, prompt_length = qa.batch_for_option(item, acquired, evidence, int(option), renderer=0)
            _policy_mode(backbone, False)
            with torch.no_grad():
                old_pair = _pair_log_probs(backbone, batch)
                sampled = torch.multinomial(old_pair.exp(), int(cfg["group_size"]), replacement=True)
                target = int(labels_by_id[item["id"]][option])
                rewards = (sampled == target).float()
                std = rewards.std(unbiased=False)
                entry = {"group": group_index+1, "record_id": item["id"], "option_index": int(option),
                         "acquired": list(acquired), "prompt_tokens": prompt_length,
                         "sampled_label_tokens": [int(backbone.label_ids[i]) for i in sampled.tolist()],
                         "reward_mean": float(rewards.mean()), "reward_std": float(std)}
                if float(std) == 0.:
                    history.append(entry | {"status": "skipped_zero_reward_std", "updates": updates})
                    save(group_index+1)
                    print("emer_grpo: "+json.dumps(history[-1]), flush=True)
                    del batch, old_pair, sampled, rewards
                    continue
                advantages = (rewards-rewards.mean())/(std+1e-6)
                old_log_probs = old_pair[sampled, None].detach()
                with _reference(backbone):
                    reference_pair = _pair_log_probs(backbone, batch)
                reference_log_probs = reference_pair[sampled, None].detach()
            _policy_mode(backbone, bool(cfg["gradient_checkpointing"]))
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                current_pair = _pair_log_probs(backbone, batch)
                current_log_probs = current_pair[sampled, None]
                loss, diagnostics = token_grpo_loss(current_log_probs, old_log_probs, reference_log_probs,
                    torch.ones_like(current_log_probs), advantages[:, None],
                    float(cfg["clip_epsilon"]), float(cfg["kl_beta"]))
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, float(cfg["max_grad_norm"]))
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Non-finite EMER GRPO LoRA gradient; optimizer update was not applied")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            history.append(entry | {"status": "updated", "updates": updates, "loss": float(loss.detach()),
                                    "gradient_norm": float(norm), **diagnostics})
            save(group_index+1)
            print("emer_grpo: "+json.dumps(history[-1]), flush=True)
            del batch, old_pair, sampled, rewards, reference_pair, current_pair, current_log_probs, loss
    finally:
        optimizer.zero_grad(set_to_none=True)
        backbone.model.eval().requires_grad_(False)
        backbone.discriminator_training = False
        if cfg["gradient_checkpointing"]:
            backbone.model.gradient_checkpointing_disable()
    final = _state(backbone)
    delta = math.sqrt(sum(float((final[key]-initial[key]).double().square().sum()) for key in final))
    report = {"algorithm": ALGORITHM, "signature": expected, "actual_updates": updates,
              "attempted_groups": len(history), "skipped_zero_reward_std": sum(
                  row["status"] == "skipped_zero_reward_std" for row in history),
              "parameter_delta_norm": delta, "train_ids": sorted(ids), "module_names": names,
              "config": cfg, "history": history,
              "completion": "one_native_binary_label_token", "temperature": 1.,
              "policy_basis": "same_native_two_label_conditional_softmax_for_old_ref_current_and_inference",
              "reference": "same_frozen_foundation_with_only_new_lora_disabled",
              "reader_frozen": True, "foundation_frozen": True,
              "requested_updates_completed": updates == int(cfg["max_updates"])}
    if updates == 0 or delta == 0:
        raise RuntimeError("No real EMER GRPO parameter update; inspect training.pt instead of treating it as trained")
    fingerprint = grpo.signature((ALGORITHM, cfg, final))
    bundle = {"schema": "emer-qa-discriminator-v1", "algorithm": ALGORITHM, "signature": expected,
              "fingerprint": fingerprint, "config": cfg, "state_dict": final,
              "actual_updates": updates, "parameter_delta_norm": delta, "module_names": names,
              "report": report}
    grpo.save_checkpoint(final_path, bundle)
    backbone.discriminator_fingerprint = fingerprint
    backbone.discriminator_bundle = bundle
    return report
