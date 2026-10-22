"""EMER multi-label QA adaptation of the EGA acquisition Controller.

This module never opens media or calls a language model.  Its inputs are cached
predictions and records from *already observed* sources.  Candidate inputs are
restricted to acquisition metadata.  Reference labels are accepted only by the
paired-teacher builder and are never part of inference state encoding.
"""
from __future__ import annotations

import copy
import hashlib
import math
import re

import numpy as np
import torch
from torch.nn import functional as F

from ..controller import Controller, sensitivity_from_log_variance


VERSION = "emer_qa_controller_v2"
MODALITIES = ("T", "A", "V")
READER_ONTOLOGY = {
    "speaker": ["unknown", "participant", "interviewer", "other"],
    "experiencer": ["unknown", "self", "other"],
    "time": ["unknown", "current", "past", "future"],
    "polarity": ["unknown", "affirmed", "denied"],
    "degree": ["unknown", "mild", "moderate", "severe"],
    "status": ["unknown", "self_report", "other_report", "hypothetical",
               "question_or_request", "interpretation"],
}
DEFAULTS = {
    "seed": 27,
    "epochs": 100,
    "patience": 10,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "sensitivity_weight": 0.1,
    "response_weight": 0.1,
    "f1_weight": 1.0,
    "brier_weight": 0.1,
    "policy_weight": 0.2,
    "policy_temperature": 0.1,
    "sensitivity_lambda": 0.1,
    "cost_weight": 0.01,
    "duration_unit_seconds": 10.0,
    "cost_per_view": 1.0,
    "max_grad_norm": 1.0,
    "min_delta": 0.000001,
    "device": "cpu",
}


def settings(config=None):
    result = DEFAULTS | (config or {})
    for key in ("learning_rate", "duration_unit_seconds", "max_grad_norm", "policy_temperature"):
        if not math.isfinite(float(result[key])) or float(result[key]) <= 0:
            raise ValueError(f"EMER Controller {key} must be positive")
    for key in ("weight_decay", "sensitivity_weight", "response_weight",
                "sensitivity_lambda", "cost_weight", "cost_per_view", "min_delta",
                "f1_weight", "brier_weight", "policy_weight"):
        if not math.isfinite(float(result[key])) or float(result[key]) < 0:
            raise ValueError(f"EMER Controller {key} must be nonnegative")
    if int(result["epochs"]) < 1 or int(result["patience"]) < 1:
        raise ValueError("EMER Controller needs positive epochs and patience")
    if float(result["f1_weight"])+float(result["brier_weight"]) <= 0:
        raise ValueError("EMER Controller needs a positive QA risk weight")
    return result


def _probabilities(value, renderers=False):
    result = np.asarray(value, dtype=np.float64)
    shape_ok = (result.ndim == 2 and result.shape[0] == 3 and result.shape[1] > 0
                if renderers else result.ndim == 1 and len(result) > 0)
    if (not shape_ok or not np.isfinite(result).all()
            or np.any((result < 0) | (result > 1))):
        raise ValueError("Expected finite independent QA probabilities in [0, 1]")
    return result


def paired_targets(before, after, labels, config=None):
    """Matched three-renderer gain, variance and null-response QA supervision.

    Each column is one independent yes/no option probability; rows must retain
    the same option ordering across the three semantic-equivalent renderers.
    """
    left, right = (_probabilities(x, True) for x in (before, after))
    target = np.asarray(labels, dtype=np.float64)
    if (left.shape != right.shape or target.shape != (left.shape[1],)
            or not np.isfinite(target).all() or not np.isin(target, (0, 1)).all()):
        raise ValueError("QA teacher needs matched [3, K] probabilities and K binary labels")
    cfg = settings(config)
    brier0, brier1 = [((x-target[None, :])**2).mean(1) for x in (left, right)]
    f1 = []
    for values in (left, right):
        positive = values >= 0.5
        numerator = 2*(positive & (target[None, :] == 1)).sum(1)
        denominator = positive.sum(1)+target.sum()
        f1.append(np.divide(numerator, denominator, out=np.ones(3, dtype=np.float64),
                            where=denominator > 0))
    loss0, loss1 = [float(cfg["f1_weight"])*(1-score)+float(cfg["brier_weight"])*brier
                    for score, brier in zip(f1, (brier0, brier1))]
    gains = loss0-loss1
    return {"gain": float(gains[0]), "variance": float(np.var(gains, ddof=1)),
            "response": (loss0-loss0[0]).tolist(),
            "before_losses": loss0.tolist(), "after_losses": loss1.tolist(),
            "before_risks": loss0.tolist(), "after_risks": loss1.tolist(),
            "before_brier_losses": brier0.tolist(), "after_brier_losses": brier1.tolist(),
            "before_sample_f1": f1[0].tolist(), "after_sample_f1": f1[1].tolist(),
            "risk_weights": {key: float(cfg[key]) for key in ("f1_weight", "brier_weight")},
            "loss_basis": "sample_f1_error_plus_weighted_mean_option_brier_v2"}


def _hash_text(text):
    vector = np.zeros(64, dtype=np.float32)
    for token in re.findall(r"[\u3400-\u9fff]|[^\W_]+", str(text).casefold()):
        value = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        vector[int.from_bytes(value[:2], "little") % 64] += 1.0 if value[2] % 2 else -1.0
    return vector/max(1.0, float(np.linalg.norm(vector)))


def _metadata(view, observations, candidate=False):
    """Seven public metadata coordinates; V is explicitly video, never F."""
    modality = view.get("modality")
    if modality not in (("A", "V") if candidate else MODALITIES):
        raise ValueError("EMER acquisition candidates must be A/V; observations must be T/A/V")
    start, stop = float(view.get("start", 0.0)), float(view.get("stop", 0.0))
    if not np.isfinite([start, stop]).all() or not 0 <= start <= stop:
        raise ValueError("Acquisition metadata needs finite nonnegative start/stop seconds")
    event = view.get("event")
    shared = event is not None and any(v.get("event") == event for v in observations)
    return np.asarray([float(modality == m) for m in MODALITIES]
                      + [start/3600.0, (stop-start)/20.0,
                         float(bool(view.get("available", True))), float(shared)], dtype=np.float32)


def _reader_vector(record, view):
    """Structured soft fields and real confidences; no categorical text hash.

    The shared Reader's 64-value layout is retained, while recognizing the
    native grounded Reader's distinct probability provenance. A deterministic
    span has no learned span confidence; its missing coordinate stays zero and
    its availability flag is false.
    """
    if record.get("relation_probability_basis") not in (
            "reader_relation_softmax_v1", "grounded_context_native_choice_softmax_v1"):
        raise ValueError("Reader state requires declared native relation probabilities")
    probabilities = record.get("relation_probabilities")
    supported = record.get("relation_probability_supported")
    if (not isinstance(probabilities, dict) or set(probabilities) != set(READER_ONTOLOGY)
            or not isinstance(supported, (list, tuple)) or len(set(supported)) != len(supported)
            or not set(supported).issubset(READER_ONTOLOGY)):
        raise ValueError("Structured Reader state requires six complete relation distributions")
    def confidence(value):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
                or not math.isfinite(float(value)) or not 0 <= float(value) <= 1):
            raise ValueError("Reader state confidence must lie in [0, 1]")
        return float(value)
    values = []
    for field in sorted(READER_ONTOLOGY):
        categories = sorted(READER_ONTOLOGY[field])
        payload = probabilities[field]
        if not isinstance(payload, dict) or set(payload) != set(categories):
            raise ValueError("Reader probability categories differ from the fixed EMER ontology")
        row = [confidence(payload[category]) for category in categories]
        if not np.isclose(sum(row), 1.0, rtol=0.0, atol=1e-5):
            raise ValueError("Reader relation probabilities must sum to one")
        if field not in supported:
            expected = [float(category == "unknown") for category in categories]
            if not np.allclose(row, expected, rtol=0.0, atol=1e-6):
                raise ValueError("Unsupported Reader relations must remain unknown")
        values.extend(row)
    relation_confidence = record.get("relation_confidence", {})
    values.extend(confidence(relation_confidence.get(field, 0.0)) for field in sorted(READER_ONTOLOGY))
    diagnostics = ("presence_confidence", "span_confidence", "confidence")
    values.extend(confidence(record.get(key, 0.0)) for key in diagnostics)
    capacity = record.get("reader_positions", view.get("reader_positions"))
    if (isinstance(capacity, (bool, np.bool_)) or not isinstance(capacity, (int, np.integer))
            or capacity <= 1):
        raise ValueError("Structured Reader state needs the actual indexed source capacity")
    capacity = int(capacity)-1  # Exclude the Reader's metadata position.
    for span in (record, record.get("predicted_span", record)):
        start, stop = span["start"], span["stop"]
        if (isinstance(start, (bool, np.bool_)) or isinstance(stop, (bool, np.bool_))
                or not isinstance(start, (int, np.integer)) or not isinstance(stop, (int, np.integer))
                or not 0 <= start < stop <= capacity):
            raise ValueError("Reader span exceeds the actual indexed source capacity")
        values.extend((float(start)/capacity, float(stop)/capacity))
    values.append(float(all(key in record for key in diagnostics)))
    # Preserve the per-coordinate availability so an absent span score is not
    # interpreted as a measured zero-confidence localization.
    values.extend(float(key in record) for key in diagnostics)
    if len(values) > 63:
        raise ValueError("Structured EMER Reader state exceeds its fixed width")
    vector = np.zeros(64, dtype=np.float32)
    vector[:len(values)], vector[63] = values, 1.0
    return vector


def encode_state(observations, evidence, controller_context, probabilities,
                 remaining_fraction=1.0, remove_reader=False):
    """Encode observed inputs only, preserving the original Controller widths.

    Observation fields: id, modality, start, stop, optional event/text.  A/V
    payloads are represented by their *cached observed* discriminator context;
    this encoder does not touch their paths. Reader rows contain structured
    field distributions and confidence. A separate QA row retains all six
    option probabilities and their decision margins; no reference is accepted.
    """
    observations = list(observations)
    if not observations:
        raise ValueError("EMER state requires at least one observed source")
    ids = [str(v["id"]) for v in observations]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate observed source ids")
    lookup = dict(zip(ids, observations))
    rows, row_labels = [], []
    for view in observations:
        # Only supplied observed text enters; candidate text is never encoded.
        vector = _hash_text(view.get("text", ""))
        rows.append(np.concatenate([vector, _metadata(view, observations)]))
        row_labels.append("observed:"+str(view["modality"]))
    reader_rows = []
    if not remove_reader:
        for record in evidence or []:
            source_id = str(record.get("source_id", ""))
            if source_id not in lookup:
                raise ValueError("Reader record refers to an unobserved source")
            view = lookup[source_id]
            vector = _reader_vector(record, view)
            reader_rows.append(len(rows))
            rows.append(np.concatenate([vector, _metadata(view, observations)]))
            row_labels.append("reader:structured_soft")
    context_rows = []
    if controller_context is not None:
        declared = controller_context.get("observed_view_ids")
        if declared is None or len(declared) != len(ids) or set(declared) != set(ids):
            raise ValueError("Cached discriminator context must match exactly the observed sources")
        values = np.asarray([controller_context.get("mean"), controller_context.get("max")], dtype=np.float32)
        if values.shape != (2, 64) or not np.isfinite(values).all():
            raise ValueError("Cached discriminator context requires two finite 64-dimensional vectors")
        for vector in values:
            context_rows.append(len(rows))
            rows.append(np.concatenate([vector, np.zeros(7, dtype=np.float32)]))
            row_labels.append("discriminator_context:"+("mean" if len(context_rows) == 1 else "max"))
    p = _probabilities(probabilities)
    if p.shape != (6,):
        raise ValueError("EMER QA state requires six independent option probabilities")
    clipped = np.clip(p, 1e-7, 1-1e-7)
    entropy = float(-(clipped*np.log2(clipped)+(1-clipped)*np.log2(1-clipped)).mean())
    qa_vector = np.zeros(64, dtype=np.float32)
    qa_vector[:6], qa_vector[6:12], qa_vector[12:18] = p, np.abs(p-0.5), p >= 0.5
    qa_vector[18], qa_vector[62] = entropy, 1.0
    qa_rows = [len(rows)]
    rows.append(np.concatenate([qa_vector, np.zeros(7, dtype=np.float32)]))
    row_labels.append("qa:option_probabilities_absolute_margins_predicted_labels_entropy")
    fraction = float(remaining_fraction)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Remaining budget fraction must lie in [0, 1]")
    rows = np.asarray(rows, dtype=np.float32)
    content_norms = np.linalg.norm(rows[:, :64], axis=1)
    # A large native max-context norm must not dominate the shared row encoder
    # simply because its features use different units. Preserve public metadata
    # and apply one fixed, non-fitted rule to every 64-value content row.
    rows[:, :64] /= np.maximum(1.0, content_norms)[:, None]
    return {"x": rows,
            "state": np.asarray([entropy, 1.0, fraction], dtype=np.float32),
            "reader_row_indices": reader_rows, "context_row_indices": context_rows,
            "qa_row_indices": qa_rows, "row_labels": row_labels,
            "content_l2_before_normalization": content_norms.tolist(),
            "content_normalization": "row_semantic64_l2_clamp_min_one_metadata_unscaled_v1",
            "observed_view_ids": ids,
            "observations": [{key: v[key] for key in ("id", "modality", "start", "stop", "event", "available")
                              if key in v} for v in observations],
            "representation": VERSION, "reader_state_basis": "structured_soft_with_confidence_availability",
            "state_basis": "observed_sources_structured_reader_qa_option_row_entropy_remaining_budget_v2"}


def candidate_cost(candidate, config=None):
    """Fixed metadata-only cost in declared duration units, not GPU wall time."""
    cfg = settings(config)
    _metadata(candidate, [], candidate=True)
    duration = float(candidate.get("stop", 0.0))-float(candidate.get("start", 0.0))
    return float(cfg["cost_weight"])*(float(cfg["cost_per_view"])
           + duration/float(cfg["duration_unit_seconds"]))


def make_pair(record_id, split, encoded_state, candidate, before, after, labels, config=None):
    """Create a labeled teacher pair outside the model-facing inference API."""
    if split not in ("train", "dev"):
        raise ValueError("Paired teacher construction is restricted to train/dev")
    x, state = _state_arrays(encoded_state)
    target = paired_targets(before, after, labels, config)
    return {"record_id": str(record_id), "split": split, "x": x.copy(), "state": state.copy(),
            "observed_view_ids": list(encoded_state["observed_view_ids"]),
            "row_labels": list(encoded_state.get("row_labels", [])),
            "reader_row_indices": list(encoded_state.get("reader_row_indices", [])),
            "context_row_indices": list(encoded_state.get("context_row_indices", [])),
            "qa_row_indices": list(encoded_state.get("qa_row_indices", [])),
            "content_normalization": encoded_state.get("content_normalization"),
            "candidate": _metadata(candidate, encoded_state.get("observations", []), candidate=True),
            "action_id": str(candidate["id"]), "cost": candidate_cost(candidate, config),
            "before_probabilities": np.asarray(before).tolist(),
            "after_probabilities": np.asarray(after).tolist(), "labels": list(labels),
            "representation": VERSION, **target}


def _state_arrays(encoded_state):
    x, state = np.asarray(encoded_state["x"], dtype=np.float32), np.asarray(encoded_state["state"], dtype=np.float32)
    if (x.ndim != 2 or x.shape[1] != 71 or not len(x) or state.shape != (3,)
            or not np.isfinite(x).all() or not np.isfinite(state).all()):
        raise ValueError("Controller state requires nonempty [N, 71] rows and a 3-vector")
    return x, state


def _predict_pair(model, pair, device):
    x, state = _state_arrays(pair)
    candidate = np.asarray(pair["candidate"], dtype=np.float32)
    if candidate.shape != (7,) or not np.isfinite(candidate).all():
        raise ValueError("Controller teacher candidate must contain seven finite metadata values")
    return model(torch.as_tensor(x, device=device),
        torch.as_tensor(state, device=device), torch.as_tensor(candidate, device=device), normalized=True)


def _pair_loss(model, pair, cfg, device, scale, prediction=None):
    gain, log_variance, response = (_predict_pair(model, pair, device)
                                    if prediction is None else prediction)
    target_gain = gain.new_tensor(float(pair["gain"])/scale)
    # Match sqrt(variance + epsilon), avoiding sqrt'(0) on zero-variance pairs.
    target_std = gain.new_tensor(math.sqrt(float(pair["variance"])/scale**2+1e-6))
    target_response = torch.as_tensor(pair["response"], dtype=gain.dtype, device=device)/scale
    std = (0.5*log_variance.clamp(-30, 30)).exp()
    return ((gain-target_gain).square()+float(cfg["sensitivity_weight"])*(std-target_std).square()
            +float(cfg["response_weight"])*F.mse_loss(response, target_response))


def _state_groups(pairs):
    groups = {}
    for pair in pairs:
        observed = pair.get("observed_view_ids")
        if not isinstance(observed, list) or not observed or len(set(observed)) != len(observed):
            raise ValueError("Controller policy teachers require explicit observed source IDs")
        key = (pair["record_id"], tuple(sorted(observed)))
        group = groups.setdefault(key, [])
        if group:
            first = group[0]
            if (not np.array_equal(first["x"], pair["x"])
                    or not np.array_equal(first["state"], pair["state"])
                    or not np.array_equal(first["before_probabilities"], pair["before_probabilities"])
                    or first["labels"] != pair["labels"]
                    or pair["action_id"] in {p["action_id"] for p in group}):
                raise ValueError("Action teachers in one observed state must share exactly its inputs")
        group.append(pair)
    return [sorted(group, key=lambda pair: pair["action_id"]) for group in groups.values()]


def _group_loss(model, pairs, cfg, device, scale):
    """Regress paired risks and learn the TRAIN/DEV-derived action versus STOP.

    Policy logits use the same raw risk units, cost, sensitivity and zero STOP
    utility as deployment. The target never comes from an evaluation record.
    """
    predictions = [_predict_pair(model, pair, device) for pair in pairs]
    regression = torch.stack([_pair_loss(model, pair, cfg, device, scale, prediction)
                              for pair, prediction in zip(pairs, predictions)]).mean()
    utilities = [predictions[0][0].new_zeros(())]
    teacher_utilities = [0.0]
    for pair, (gain, log_variance, _) in zip(pairs, predictions):
        std = sensitivity_from_log_variance(log_variance)
        utilities.append(scale*(gain-float(cfg["sensitivity_lambda"])*std)-float(pair["cost"]))
        teacher_utilities.append(float(pair["gain"])
            -float(cfg["sensitivity_lambda"])*math.sqrt(float(pair["variance"]))-float(pair["cost"]))
    # STOP wins a zero-utility tie; otherwise match choose()'s action-ID tie break.
    target = 0
    if max(teacher_utilities) > 0:
        target = max(range(1, len(teacher_utilities)),
                     key=lambda i: (teacher_utilities[i], pairs[i-1]["action_id"]))
    logits = torch.stack(utilities)/float(cfg["policy_temperature"])
    policy = F.cross_entropy(logits[None, :], torch.tensor([target], dtype=torch.long, device=device))
    total = regression+float(cfg["policy_weight"])*policy
    predicted = 0 if max(float(value.detach()) for value in utilities) <= 0 else max(
        range(1, len(utilities)), key=lambda i: (float(utilities[i].detach()), pairs[i-1]["action_id"]))
    return total, {"regression_loss": float(regression.detach()), "policy_loss": float(policy.detach()),
                   "action_correct": predicted == target, "stop_target": target == 0,
                   "stop_correct": (predicted == 0) == (target == 0)}


def _mean_group_report(model, groups, cfg, device, scale):
    outputs = [_group_loss(model, group, cfg, device, scale) for group in groups]
    stop_rows = [row for _, row in outputs if row["stop_target"]]
    return {"loss": float(np.mean([float(loss) for loss, _ in outputs])),
            "regression_loss": float(np.mean([row["regression_loss"] for _, row in outputs])),
            "policy_loss": float(np.mean([row["policy_loss"] for _, row in outputs])),
            "action_accuracy": float(np.mean([row["action_correct"] for _, row in outputs])),
            "stop_accuracy": float(np.mean([row["stop_correct"] for _, row in outputs])),
            "stop_recall": (float(np.mean([row["stop_correct"] for row in stop_rows])) if stop_rows else None),
            "states": len(groups), "stop_target_states": len(stop_rows)}


def fit(train_pairs, dev_pairs, config=None):
    """One fixed Adam configuration; early stopping uses joint DEV loss only."""
    cfg = settings(config)
    train, dev = list(train_pairs), list(dev_pairs)
    if not train or not dev:
        raise ValueError("EMER Controller requires disjoint train and dev teacher pairs")
    if any(p.get("split") != "train" for p in train) or any(p.get("split") != "dev" for p in dev):
        raise ValueError("Controller teacher pair split mismatch")
    train_ids, dev_ids = {p["record_id"] for p in train}, {p["record_id"] for p in dev}
    if train_ids & dev_ids:
        raise ValueError("Train and dev video records must be disjoint")
    for pair in train+dev:
        if pair.get("representation") != VERSION:
            raise ValueError("Controller teacher representation changed")
        checked = paired_targets(pair["before_probabilities"], pair["after_probabilities"], pair["labels"], cfg)
        if (not np.allclose([pair["gain"], pair["variance"]], [checked["gain"], checked["variance"]])
                or not np.allclose(pair["response"], checked["response"])
                or pair.get("risk_weights") != checked["risk_weights"]
                or pair.get("loss_basis") != checked["loss_basis"]):
            raise ValueError("Paired Controller targets do not match their cached QA predictions")
    train_groups, dev_groups = _state_groups(train), _state_groups(dev)
    # Scaling is fitted once on TRAIN; public predictions retain joint-risk units.
    values = [float(p["gain"]) for p in train]+[float(r) for p in train for r in p["response"]]
    scale = max(0.001, float(np.sqrt(np.mean(np.asarray(values)**2))))
    torch.manual_seed(int(cfg["seed"]))
    rng = np.random.default_rng(int(cfg["seed"]))
    device = torch.device(cfg["device"])
    model = Controller(target_normalization="source_train_weighted_rms_v1", target_scale=scale).to(device)
    with torch.no_grad():
        model.gain[-1].bias.fill_(float(np.mean([p["gain"] for p in train]))/scale)
        model.sensitivity[-1].bias.fill_(math.log(float(np.mean([p["variance"] for p in train]))/scale**2+1e-6))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]),
                                 weight_decay=float(cfg["weight_decay"]))
    best, best_epoch, best_weights, bad_epochs, history = math.inf, 0, None, 0, []
    initial_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for epoch in range(1, int(cfg["epochs"])+1):
        model.train()
        train_losses, train_rows = [], []
        for index in rng.permutation(len(train_groups)):
            optimizer.zero_grad(set_to_none=True)
            loss, row = _group_loss(model, train_groups[int(index)], cfg, device, scale)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite EMER Controller training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
            optimizer.step()
            train_losses.append(float(loss.detach()))
            train_rows.append(row)
        model.eval()
        with torch.no_grad():
            dev_report = _mean_group_report(model, dev_groups, cfg, device, scale)
        dev_loss = dev_report["loss"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "dev_loss": dev_loss,
                        "train_regression_loss": float(np.mean([r["regression_loss"] for r in train_rows])),
                        "train_policy_loss": float(np.mean([r["policy_loss"] for r in train_rows])),
                        "dev": dev_report})
        if dev_loss < best-float(cfg["min_delta"]):
            best, best_epoch, bad_epochs = dev_loss, epoch, 0
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
        if bad_epochs >= int(cfg["patience"]):
            break
    if best_weights is None:
        raise RuntimeError("Controller did not produce a finite development checkpoint")
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        final_dev = _mean_group_report(model, dev_groups, cfg, device, scale)
    delta = math.sqrt(sum(float((best_weights[k]-initial_weights[k]).double().square().sum())
                          for k in best_weights))
    return model, {"version": VERSION, "settings": cfg, "best_epoch": best_epoch,
                   "best_dev_loss": best, "history": history, "train_pairs": len(train),
                   "dev_pairs": len(dev), "train_videos": sorted(train_ids), "dev_videos": sorted(dev_ids),
                   "train_states": len(train_groups), "dev_states": len(dev_groups),
                   "development": final_dev, "dev_action_accuracy": final_dev["action_accuracy"],
                   "dev_stop_accuracy": final_dev["stop_accuracy"],
                   "target_scale_train_only": scale, "parameter_delta_norm": delta,
                   "cost_basis": "fixed_per_view_plus_audio_or_video_duration_v1",
                   "loss_basis": "sample_f1_error_plus_weighted_mean_option_brier_v2",
                   "policy_basis": "paired_teacher_net_utility_action_or_zero_stop_cross_entropy_v1",
                   "selection": "single_fixed_configuration_joint_dev_loss_early_stopping",
                   "model_class": "tmm.controller.Controller", "candidate_conditioning": "none"}


def _score_candidates(model, encoded_state, candidates, cfg):
    x, state = _state_arrays(encoded_state)
    device = next(model.parameters()).device
    observed = set(encoded_state.get("observed_view_ids", []))
    scored = []
    model.eval()
    with torch.no_grad():
        for candidate in candidates:
            identity = str(candidate["id"])
            if identity in observed or not candidate.get("available", True):
                continue
            metadata = _metadata(candidate, encoded_state.get("observations", []), candidate=True)
            gain, log_variance, _ = model(torch.as_tensor(x, device=device),
                torch.as_tensor(state, device=device), torch.as_tensor(metadata, device=device))
            gain, log_variance = float(gain), float(log_variance)
            variance = max(0.0, math.exp(np.clip(log_variance, -30.0, 30.0))-1e-6)
            sensitivity = math.sqrt(variance)
            cost = candidate_cost(candidate, cfg)
            q = gain-float(cfg["sensitivity_lambda"])*sensitivity-cost
            if not np.isfinite([gain, variance, cost, q]).all():
                raise RuntimeError("Non-finite EMER acquisition utility")
            scored.append({"id": identity, "modality": candidate["modality"],
                           "gain": gain, "variance": variance, "sensitivity": sensitivity,
                           "sensitivity_penalty": float(cfg["sensitivity_lambda"])*sensitivity,
                           "log_variance": log_variance, "cost": cost, "q": q,
                           "start": float(candidate.get("start", 0)), "stop": float(candidate.get("stop", 0))})
    best = max(scored, key=lambda row: (row["q"], row["id"])) if scored else None
    stop = best is None or best["q"] <= 0
    return {"action": "STOP" if stop else best["id"], "stop": stop,
            "q": None if best is None else best["q"], "candidates": scored,
            "utility_basis": "sample_f1_error_plus_weighted_mean_option_brier_v2",
            "stop_reason": "no_candidates" if best is None else "nonpositive_utility" if stop else None}


def choose(model, encoded_state, candidates, config=None, reader_diagnostic=True):
    """Choose one metadata-only action or STOP; optional local Reader diagnostic.

    The feature-removal result reuses the fitted model on the same observed
    state.  It is not a retrained ablation, trajectory or language-model output.
    """
    cfg, candidates = settings(config), list(candidates)
    result = _score_candidates(model, encoded_state, candidates, cfg)
    rows = list(encoded_state.get("reader_row_indices", []))
    if reader_diagnostic and rows:
        reduced = copy.copy(encoded_state)
        x, _ = _state_arrays(encoded_state)
        if len(set(rows)) != len(rows) or any(type(i) is not int or not 0 <= i < len(x) for i in rows):
            raise ValueError("Invalid Reader row indices for local diagnostic")
        reduced["x"] = np.delete(x, rows, axis=0)
        reduced["reader_row_indices"] = []
        removed = _score_candidates(model, reduced, candidates, cfg)
        result["reader_feature_removal"] = {**removed, "decision_changed": result["action"] != removed["action"],
            "scope": "Same observed state and fitted Controller; Reader rows removed locally, no retraining or new model calls."}
    return result
