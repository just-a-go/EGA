#!/usr/bin/env bash
set -euo pipefail
# EGA full training/evaluation: modma / PM. One independent experiment.
# CLI: --check --stage train|evaluate|all --resume --gpu N --seed N
# Reader/controller/adapter use all eligible training participants; no smoke limits.
# Requires source data and verified participant groups; missing prerequisites remain blocked.
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="/data0/pretrained/tmm_phi4_deps:${PYTHONPATH:-}"
cd "$PROJECT_DIR"
exec "${TMM_PYTHON:-/data0/pretrained/env_humanomniv2/bin/python}" -B scripts/ega_run.py --settings - "$@" <<'SETTINGS'
{
  "gpu": 7,
  "seed": 27,
  "cpu_threads": 4,
  "budget": 8,
  "protocol": {
    "max_input_tokens": 4096,
    "max_output_tokens": 128,
    "max_evidence_tokens": 1024
  },
  "reader": {
    "enabled": true,
    "epochs": 100,
    "patience": 10,
    "learning_rate": 0.001,
    "consistency_weight": 0.1,
    "presence_thresholds": [
      0.7
    ],
    "max_positions": 2048,
    "audio_frame_seconds": 0.1,
    "loss_weights": {
      "presence": 1.0,
      "start": 1.0,
      "stop": 1.0,
      "speaker": 1.0,
      "experiencer": 1.0,
      "time": 1.0,
      "polarity": 1.0,
      "degree": 1.0,
      "status": 1.0
    },
    "annotations": "manifests/reader/items.jsonl",
    "ontology": {
      "speaker": [
        "unknown",
        "participant",
        "interviewer",
        "other"
      ],
      "experiencer": [
        "unknown",
        "self",
        "other"
      ],
      "time": [
        "unknown",
        "current",
        "past",
        "future"
      ],
      "polarity": [
        "unknown",
        "affirmed",
        "denied"
      ],
      "degree": [
        "unknown",
        "mild",
        "moderate",
        "severe"
      ],
      "status": [
        "unknown",
        "self_report",
        "measurement",
        "interpretation"
      ]
    },
    "max_windows": null
  },
  "controller": {
    "prefix_lengths": [
      0,
      1,
      3
    ],
    "actions_per_state": 3,
    "epochs": 100,
    "patience": 10,
    "learning_rate": 0.001,
    "alpha": 0.1,
    "beta": 0.1,
    "policy_lambda": 0.01,
    "policy_gamma": 0.1,
    "budget": 8,
    "grpo": {
      "enabled": false,
      "weight": 0.0,
      "supervised_weight": 1.0,
      "group_size": 16,
      "temperature": 1.0,
      "clip_epsilon": 0.2,
      "kl_beta": 0.04,
      "updates_per_group": 2,
      "max_grad_norm": 1.0,
      "max_steps": null
    },
    "max_teacher_states": null,
    "max_teacher_pairs": null
  },
  "adapter": {
    "seed": 27,
    "learning_rate": 0.001,
    "weight_decay": [
      0.01
    ],
    "epochs": 100,
    "patience": 10
  },
  "screening": {
    "enabled": true,
    "constrained_json": true,
    "max_new_tokens": 128,
    "disorder_keys": [
      "depression",
      "anxiety",
      "ptsd"
    ],
    "accuracy_weight": 1.0,
    "structure_weight": 0.1,
    "safety_weight": 0.1,
    "target_disorder": "depression"
  },
  "parameter_provenance": "Token-policy discriminator token GRPO using engineering LoRA/reward settings; supervised Reader and paired-regression Controller; development checkpoint selection.",
  "task": "modma",
  "model": "PM",
  "pretrained_weights": "/data0/pretrained/Phi-4-multimodal-instruct",
  "disable_compile": false,
  "run_kind": "full_training",
  "stage": "all",
  "config": "configs/experiments.json",
  "training_task": "modma",
  "reader_sources": [
    "daic",
    "modma"
  ],
  "evaluation": {
    "protocol": "nested",
    "split": "test",
    "group_by": "participant",
    "outer_folds": 5,
    "inner_folds": 4,
    "stratified": true
  },
  "acquisition_limit": 8,
  "discriminator": {
    "enabled": true,
    "algorithm": "token_grpo_lora",
    "learning_rate": 1e-05,
    "epochs": 3,
    "patience": 1,
    "group_size": 4,
    "temperature": 0.8,
    "top_p": 1.0,
    "top_k": 0,
    "clip_epsilon": 0.2,
    "kl_beta": 0.04,
    "max_new_tokens": 128,
    "gradient_accumulation_groups": 4,
    "updates_per_group": 2,
    "max_grad_norm": 1.0,
    "gradient_checkpointing": true,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "max_steps": null,
    "prefix_lengths": [
      0,
      1,
      3
    ],
    "target_modules": [
      "qkv_proj"
    ],
    "weight_decay": 0.0,
    "constrained_json": false,
    "repetition_penalty": 1.0
  }
}
SETTINGS
