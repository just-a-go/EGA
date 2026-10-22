#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
exec "${TMM_PYTHON:-/data0/pretrained/env_humanomniv2/bin/python}" -B -u scripts/emer_run.py --settings - "$@" <<'SETTINGS'
{
  "gpu": 2,
  "seed": 27,
  "cpu_threads": 4,
  "max_image_pixels": 65536,
  "asr_max_new_tokens": 384,
  "model": {
    "name": "Qwen2.5-Omni-7B",
    "kind": "omni25",
    "path": "/data0/pretrained/Qwen2.5-Omni-7B-Thinker",
    "thinking": false,
    "attention": "sdpa",
    "disable_compile": true
  },
  "protocol": {"max_input_tokens": 4096, "max_output_tokens": 384, "max_evidence_tokens": 1024},
  "reader": {
    "mode": "grounded_context", "max_candidates": 8, "max_records": 2,
    "relevance_threshold": 0.5, "relation_confidence_threshold": 0.6
  },
  "discriminator": {
    "seed": 27, "group_size": 32, "max_updates": 24, "max_groups": 96,
    "learning_rate": 0.00001, "temperature": 1.0, "clip_epsilon": 0.2,
    "kl_beta": 0.01, "lora_rank": 8, "lora_alpha": 16,
    "target_modules": ["q_proj", "v_proj"], "max_grad_norm": 1.0,
    "gradient_checkpointing": true
  },
  "controller": {
    "seed": 27, "epochs": 100, "patience": 10, "learning_rate": 0.001,
    "weight_decay": 0.0001, "max_grad_norm": 1.0, "min_delta": 0.000001,
    "sensitivity_weight": 0.1, "response_weight": 0.1, "sensitivity_lambda": 0.1,
    "f1_weight": 1.0, "brier_weight": 0.1, "policy_weight": 0.2, "policy_temperature": 0.1,
    "cost_weight": 0.002, "duration_unit_seconds": 10.0, "cost_per_view": 1.0,
    "device": "cpu"
  }
}
SETTINGS
