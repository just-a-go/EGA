#!/usr/bin/env bash
set -euo pipefail
# EATD depression / Qwen2.5-Omni. All training parameters are declared below.
# Fresh Reader, discriminator LoRA and Controller training; no prior run is required.
# Options: --gpu N --seed N --stage train|evaluate|all --resume --check --show-settings
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="/data0/pretrained/tmm_phi4_deps:${PYTHONPATH:-}"
cd "$PROJECT_DIR"
exec "${TMM_PYTHON:-/data0/pretrained/env_humanomniv2/bin/python}" -B scripts/ega_run.py --settings - "$@" <<'SETTINGS'
{
  "gpu": 3,
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
    "max_windows": null,
    "relation_confidence_threshold": 0.7,
    "relation_class_balance": "sqrt_inverse_frequency",
    "selection_metric": "span_relation_error",
    "record_abstention_cost": 0.5,
    "memory_modalities": "supervised",
    "relation_conditioning": "predicted_span_mean",
    "boundary_likelihood": "global_legal_joint_v1",
    "speaker_normalization": "ontology_casefold_v1",
    "training_mode": "participant_minibatch_v1",
    "participant_batch_size": 16
  },
  "controller": {
    "state_representation": "frozen_discriminator_context",
    "candidate_conditioning": "observed_event",
    "action_filter": "aligned_audio",
    "metadata_normalization": "source_clipped_minmax",
    "gain_loss": {
      "mode": "source_cut_centered_squared"
    },
    "ranking_loss": {
      "enabled": true,
      "weight": 0.1,
      "temperature": 0.1
    },
    "seed_prediction_anchor": "plain_until_acquisition",
    "prefix_lengths": [
      0,
      1,
      3
    ],
    "actions_per_state": 3,
    "teacher_augmentation": {
      "mode": "disabled"
    },
    "epochs": 100,
    "patience": 10,
    "learning_rate": 0.001,
    "alpha": 0.1,
    "sensitivity_loss": "std_mse",
    "decision_loss": {
      "enabled": false,
      "mode": "binary_stop",
      "weight": 0.0,
      "temperature": 0.1,
      "class_balance": "stop_acquire",
      "candidate_ranking_weight": 0.0
    },
    "beta": 0.1,
    "policy_lambda": 0.01,
    "policy_gamma": 0.1,
    "selection_metric": "initial_utility_regret",
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
    "max_teacher_pairs": null,
    "acquisition_limit": 8,
    "teacher_sampling": {
      "mode": "aligned_audio"
    },
    "score_calibration": {
      "mode": "source_audio_cut_transport"
    },
    "training_state_weighting": "initial_later_balanced",
    "state_evidence": "observed_reader",
    "state_evidence_encoding": "structured_soft",
    "reader_diagnostics": true,
    "state_filter": "seed_aligned_audio",
    "teacher_rollout": {
      "mode": "seed_aligned_audio"
    },
    "budget_normalization": "source_train_seed_median",
    "participant_order": "seeded_epoch_shuffle",
    "target_normalization": "none"
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
    "constrained_json": false,
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
  "task": "eatd",
  "model": "QO",
  "pretrained_weights": "/data0/pretrained/Qwen2.5-Omni-7B-Thinker",
  "disable_compile": false,
  "run_kind": "full_training",
  "config": "configs/experiments.json",
  "training_task": "daic",
  "reader_sources": [
    "daic",
    "mmpsy_depression"
  ],
  "evaluation": {
    "protocol": "transfer",
    "split": "test",
    "group_by": "participant",
    "target_adaptation": false
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
      "q_proj",
      "v_proj"
    ],
    "weight_decay": 0.0,
    "constrained_json": false,
    "repetition_penalty": 1.0,
    "prompt_modes": [
      "plain",
      "reader"
    ]
  },
  "calibration": {
    "mode": "joint_source_dev",
    "reader_minimum_brier_gain": 0.0,
    "threshold_objective": "seed_shared_source_dev",
    "reader_gate_loss": "controller_gain_loss"
  },
  "parameter_provenance": "Train the Reader, discriminator LoRA and Controller afresh from source supervision. Reader training uses DAIC-WOZ and the fixed MMPsy depression TRAIN/DEV partitions; discriminator and Controller training, calibration and selection use DAIC-WOZ. EATD79 is used only for evaluation in this run.",
  "stage": "all",
  "reader_refit": false
}
SETTINGS
