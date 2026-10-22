"""EGA paired gain supervision and budget-aware evidence acquisition."""
import copy
import hashlib
import itertools
import json
import random
import re
import time

import numpy as np
import torch
from torch import nn

from .support.data import ObservationError, canonical, digest, metadata, pool, seed_views
from .support import grpo
from .support.screening import record_reward as screening_record_reward
from .support.models import CONTROLLER_CONTEXT_BASIS, feature_vector, waveform
from .support.controller_candidates import (action_filter_mode, candidate_action_allowed,
                                    candidate_conditioning_mode, encode_candidate_mask, validate_candidate_mask)
from .support.budget_units import (budget_normalization_mode, fit_budget_profile,
                           rebase_budget_examples, budget_reference)


STATE_REPRESENTATION_VERSION = 1


def controller_inputs(config):
    representation = config.get('state_representation', 'hash')
    anchor = config.get('seed_prediction_anchor', 'legacy_reader')
    if representation not in ('hash', 'frozen_token_embeddings', 'frozen_discriminator_context'):
        raise ValueError('Unsupported Controller state representation')
    if anchor not in ('legacy_reader', 'plain_until_acquisition'):
        raise ValueError('Unsupported seed prediction anchor')
    return representation, anchor


def controller_state_evidence_mode(config):
    mode = config.get('state_evidence', 'gated')
    if mode not in ('gated', 'observed_reader'):
        raise ValueError('Unsupported Controller state evidence mode')
    return mode


def frozen_text_representation(backbone, view):
    """Pool the Reader's frozen token representation; no Transformer forward."""
    if backbone is None:
        raise ValueError('Frozen Controller token embeddings require a backbone')
    embedding = backbone.model.get_input_embeddings()
    key = (id(embedding), getattr(embedding.weight, '_version', 0),
           hashlib.sha256(view['text'].encode('utf-8')).digest())
    cache = getattr(backbone, '_controller_text_cache', None)
    if cache is None:
        cache = backbone._controller_text_cache = {}
    if key not in cache:
        ids = backbone.tokenizer.encode(view['text'], add_special_tokens=False)
        if not ids:
            raise ObservationError('Controller received empty text tokens')
        with torch.no_grad():
            # Same embedding compression and per-token normalization as SourceEncoder.
            raw = embedding(torch.tensor(ids, device=embedding.weight.device)).float()
            rows = torch.nn.functional.adaptive_avg_pool1d(raw.unsqueeze(1), 64).squeeze(1)
            rows = rows / torch.linalg.vector_norm(rows, dim=1, keepdim=True).clamp_min(1.)
            pooled = rows.mean(0).cpu().numpy()
        if not np.isfinite(pooled).all():
            raise ObservationError('Non-finite Controller token representation')
        cache[key] = pooled / max(1., float(np.linalg.norm(pooled)))
    return cache[key].copy()


def representation(view, backbone=None, state_representation='hash'):
    out = np.zeros(64, dtype=np.float32)
    if view['modality'] == 'T':
        if state_representation in ('frozen_token_embeddings', 'frozen_discriminator_context'):
            return frozen_text_representation(backbone, view)
        # Whitespace splitting collapses an entire Chinese utterance into one hash.
        for word in re.findall(r'[\u3400-\u9fff]|[^\W_]+', view['text'].casefold()):
            h = hashlib.blake2b(word.encode(), digest_size=4).digest()
            out[int.from_bytes(h[:2], 'little') % 64] += 1 if h[2] % 2 else -1
    elif view['modality'] == 'A':
        wave = waveform(view)
        for i, piece in enumerate(np.array_split(wave, 32)):
            if len(piece):
                out[2*i:2*i+2] = [piece.mean(), np.sqrt(np.mean(piece**2))]
    else:
        x = np.concatenate(list(feature_vector(view).values()))
        for i, piece in enumerate(np.array_split(x, 64)):
            out[i] = piece.mean() if len(piece) else 0
    return out / max(1., float(np.linalg.norm(out)))


def controller_context_rows(views, controller_context):
    """Validate native context for exactly these acquired views; candidate content is absent."""
    if (not isinstance(controller_context, dict)
            or controller_context.get('basis') != CONTROLLER_CONTEXT_BASIS):
        raise RuntimeError('Controller native prompt context is missing or has an incompatible basis')
    observed = [view['id'] for view in canonical(views)]
    if controller_context.get('observed_view_ids') != observed:
        raise RuntimeError('Controller native prompt context does not match the acquired views')
    try:
        rows = np.asarray([controller_context.get('mean'), controller_context.get('max')], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise RuntimeError('Controller native prompt context contains invalid vectors') from exc
    if rows.shape != (2, 64) or not np.isfinite(rows).all():
        raise RuntimeError('Controller native prompt context requires two finite 64-value vectors')
    return np.concatenate([rows, np.zeros((2, 7), dtype=np.float32)], axis=1)


def encode_state(views, evidence=None, backbone=None, state_representation='hash', controller_context=None,
                 state_evidence_encoding='hash'):
    context_rows = (controller_context_rows(views, controller_context)
                    if state_representation == 'frozen_discriminator_context' else None)
    rows = [np.concatenate([representation(v, backbone, state_representation),
                            metadata(v, [x for x in views if x['id'] != v['id']])])
            for v in canonical(views)]
    if evidence:
        from .support.reader_state import encode_reader_record
        lookup = {v['id']: v for v in views}
        for record in evidence:
            view = lookup[record['source_id']]
            # Participant ids, labels and post-action outputs never enter the state.
            reader_config = getattr(getattr(backbone, 'reader', None), 'config', {}) or {}
            ontology = reader_config.get('ontology')
            rows.append(np.concatenate([encode_reader_record(record, view, ontology, state_evidence_encoding,
                                        speaker_normalization=reader_config.get('speaker_normalization', 'literal')),
                                        metadata(view, views)]))
    if context_rows is not None:
        rows.extend(context_rows)
    return np.stack(rows)


def fit_metadata_bounds(examples):
    """Joint location/duration support from raw TRAIN states and candidates only."""
    rows = []
    for example in examples:
        observations = np.asarray(example['x'], dtype=float)
        candidate = np.asarray(example['candidate'], dtype=float)
        if observations.ndim != 2 or observations.shape[1] != 71 or candidate.shape != (7,):
            raise ValueError('Controller metadata fitting requires 71-column states and 7-column candidates')
        rows.extend([observations[:, 67:69], candidate[None, 3:5]])
    if not rows:
        raise ValueError('Controller metadata bounds require training teacher inputs')
    values = np.concatenate(rows, axis=0)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Controller training metadata must be nonempty and finite')
    return {'source': 'training_teacher_inputs', 'lower': values.min(0).tolist(),
            'upper': values.max(0).tolist(), 'rows': len(values)}


def target_normalization_mode(config):
    mode = config.get('target_normalization', 'none')
    if mode not in ('none', 'source_train_weighted_rms_v1'):
        raise ValueError('Unsupported Controller target normalization')
    return mode


def _target_signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def validate_target_profile(profile):
    if profile is None:
        return None
    if not isinstance(profile, dict):
        raise ValueError('Controller target profile must be a dictionary')
    scale = profile.get('scale')
    if (profile.get('version') != 1
            or profile.get('mode') != 'source_train_weighted_rms_v1'
            or isinstance(scale, bool) or not isinstance(scale, (int, float))
            or not np.isfinite(scale) or scale <= 0
            or not isinstance(profile.get('counts'), dict)):
        raise ValueError('Invalid fitted Controller target profile')
    for key in ('train_target_signature', 'weight_signature', 'loss_signature'):
        value = profile.get(key)
        if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
            raise ValueError('Controller target profile lacks a source training signature')
    contents = {key: value for key, value in profile.items() if key != 'fingerprint'}
    if profile.get('fingerprint') != _target_signature(contents):
        raise ValueError('Controller target profile fingerprint changed')
    return copy.deepcopy(profile)


class Controller(nn.Module):
    def __init__(self, metadata_normalization='none', metadata_bounds=None, candidate_conditioning='none',
                 target_normalization='none', target_profile=None, target_scale=None):
        super().__init__()
        self.target_normalization = target_normalization_mode({'target_normalization': target_normalization})
        if target_profile is not None:
            target_profile = validate_target_profile(target_profile)
            if self.target_normalization != target_profile['mode']:
                raise ValueError('Controller target normalization/profile mismatch')
            if target_scale is not None and float(target_scale) != target_profile['scale']:
                raise ValueError('Controller target scale/profile mismatch')
            target_scale = target_profile['scale']
        if self.target_normalization != 'none':
            if target_scale is None or not np.isfinite(float(target_scale)) or float(target_scale) <= 0:
                raise ValueError('Controller target normalization requires a positive TRAIN scale')
            # Non-trainable; old checkpoints retain their exact original keys/shapes.
            self.register_buffer('target_scale', torch.tensor(float(target_scale), dtype=torch.float64))
        elif target_scale is not None:
            raise ValueError('Legacy Controller cannot carry a normalized target scale')
        if metadata_normalization not in ('none', 'source_clipped_minmax'):
            raise ValueError('Unsupported Controller metadata normalization')
        self.metadata_normalization = metadata_normalization
        self.candidate_conditioning = candidate_conditioning_mode({'candidate_conditioning': candidate_conditioning})
        if metadata_normalization == 'source_clipped_minmax':
            if metadata_bounds is None:
                raise ValueError('Source metadata normalization requires fitted TRAIN bounds')
            lower = torch.as_tensor(metadata_bounds['lower'], dtype=torch.float32)
            upper = torch.as_tensor(metadata_bounds['upper'], dtype=torch.float32)
            if (lower.shape != (2,) or upper.shape != (2,)
                    or not torch.isfinite(lower).all() or not torch.isfinite(upper).all()
                    or (upper < lower).any()):
                raise ValueError('Invalid fitted Controller metadata bounds')
            self.register_buffer('metadata_lower', lower.clone())
            self.register_buffer('metadata_upper', upper.clone())
        self.encoder = nn.Sequential(nn.Linear(71, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        action_width = 522 if self.candidate_conditioning == 'observed_event' else 266
        self.gain = nn.Sequential(nn.Linear(action_width, 128), nn.ReLU(), nn.Linear(128, 1))
        self.sensitivity = nn.Sequential(nn.Linear(action_width, 128), nn.ReLU(), nn.Linear(128, 1))
        self.response = nn.Sequential(nn.Linear(262, 128), nn.ReLU(), nn.Linear(128, 1))

    @classmethod
    def from_state_dict(cls, weights, target_profile=None):
        width = weights['gain.0.weight'].shape[1]
        if width not in (266, 522) or weights['sensitivity.0.weight'].shape[1] != width:
            raise ValueError('Controller checkpoint has incompatible candidate heads')
        mode = 'observed_event' if width == 522 else 'none'
        present = ['metadata_lower' in weights, 'metadata_upper' in weights]
        if any(present) and not all(present):
            raise ValueError('Controller checkpoint has incomplete metadata bounds')
        normalized = 'target_scale' in weights
        profile = validate_target_profile(target_profile)
        if profile is not None and not normalized:
            raise ValueError('Normalized target profile cannot restore a legacy Controller')
        target_mode = 'source_train_weighted_rms_v1' if normalized else 'none'
        if normalized and weights['target_scale'].shape != ():
            raise ValueError('Controller checkpoint target scale must be scalar')
        scale = float(weights['target_scale'].item()) if normalized else None
        options = dict(target_normalization=target_mode, target_profile=profile, target_scale=scale)
        if all(present):
            bounds = {'lower': weights['metadata_lower'].detach().cpu().tolist(),
                      'upper': weights['metadata_upper'].detach().cpu().tolist()}
            model = cls('source_clipped_minmax', bounds, mode, **options)
        else:
            model = cls(candidate_conditioning=mode, **options)
        model.load_state_dict(weights, strict=True)
        return model

    def normalize_metadata(self, observations, candidate):
        if self.metadata_normalization == 'none':
            return observations, candidate
        span = self.metadata_upper-self.metadata_lower
        variable = span > 0
        divisor = torch.where(variable, span, torch.ones_like(span))
        def scale(values):
            scaled = (values-self.metadata_lower)/divisor
            return torch.where(variable, scaled, torch.zeros_like(scaled)).clamp(0., 1.)
        observations, candidate = observations.clone(), candidate.clone()
        observations[:, 67:69] = scale(observations[:, 67:69])
        candidate[3:5] = scale(candidate[3:5])
        return observations, candidate

    def forward(self, observations, state, candidate, candidate_mask=None, normalized=False):
        observations, candidate = self.normalize_metadata(observations, candidate)
        hidden = self.encoder(observations)
        pooled = torch.cat([hidden.mean(0), hidden.max(0).values, state])
        joined = torch.cat([pooled, candidate])
        if self.candidate_conditioning == 'observed_event':
            if candidate_mask is None:
                raise ValueError('Event-conditioned Controller requires the observed-row candidate mask')
            mask = torch.as_tensor(candidate_mask, device=hidden.device)
            if (mask.shape != (len(hidden),) or not torch.isfinite(mask).all()
                    or not ((mask == 0) | (mask == 1)).all()):
                raise ValueError('Controller candidate mask is incompatible with observed rows')
            matched = hidden[mask.bool()]
            local = (torch.cat([matched.mean(0), matched.max(0).values]) if len(matched)
                     else hidden.new_zeros(256))
            joined = torch.cat([joined, local])
        elif candidate_mask is not None:
            raise ValueError('Legacy Controller cannot use a candidate event mask')
        renderers = torch.eye(3, device=pooled.device, dtype=pooled.dtype)
        null = torch.stack([self.response(torch.cat([pooled, renderers[r]])).squeeze() for r in range(3)])
        gain, q = self.gain(joined).squeeze(), self.sensitivity(joined).squeeze()
        if self.target_normalization != 'none' and not normalized:
            scale = self.target_scale.to(dtype=gain.dtype)
            # The public interface remains in original gain/variance/response units.
            raw_variance = sensitivity_from_log_variance(q).square()*scale.square()
            return gain*scale, torch.log(raw_variance+1e-6), null*scale
        return gain, q, null


def gain_loss_settings(config):
    supplied = config.get('gain_loss', {})
    if not isinstance(supplied, dict):
        raise ValueError('Controller gain_loss must be a configuration dictionary')
    mode = supplied.get('mode', 'raw_brier')
    if mode == 'raw_brier':
        return {'mode': mode}
    if mode != 'source_cut_centered_squared':
        raise ValueError('Unsupported Controller gain loss')
    raw_cut = supplied.get('source_cut')
    try:
        cut = float(raw_cut)
    except (TypeError, ValueError):
        raise ValueError('Source-centered gain loss requires a finite source_cut in (0, 1)') from None
    if isinstance(raw_cut, bool) or not np.isfinite(cut) or not 0 < cut < 1:
        raise ValueError('Source-centered gain loss requires a finite source_cut in (0, 1)')
    return {'mode': mode, 'source_cut': cut}


def gain_probability(probabilities, gain_loss):
    """Map prediction-scale probabilities for teacher loss without changing native logits."""
    values = np.asarray(probabilities, dtype=float)
    if gain_loss['mode'] == 'raw_brier':
        return values
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Paired teacher probabilities must be finite and in [0, 1]')
    cut = gain_loss['source_cut']
    numerator = values*(1-cut)
    return np.clip(numerator/(numerator+cut*(1-values)), 0., 1.)


def paired_targets(before, after, label, gain_loss=None):
    left, right = (np.asarray(p, dtype=float) for p in (before, after))
    if left.shape != (3,) or right.shape != (3,):
        raise ValueError('Three matched renderer predictions required')
    settings = gain_loss_settings({'gain_loss': gain_loss or {}})
    left, right = gain_probability(left, settings), gain_probability(right, settings)
    loss0, loss1 = (left-label)**2, (right-label)**2
    gains = loss0-loss1
    return float(gains[0]), float(np.var(gains, ddof=1)), loss0-loss0[0]


def _score_profile(profile):
    if profile is None:
        return None
    from .support.score_calibration import validate_profile
    validated = validate_profile(profile)
    return None if validated['mode'] == 'disabled' else validated


def calibrate_prediction(prediction, views, profile=None):
    """Transport a native score once; only actually acquired Audio selects the branch."""
    profile = _score_profile(profile)
    marker = prediction.get('score_calibration')
    if profile is None:
        if marker is not None:
            raise ValueError('A calibrated prediction requires its original score profile')
        return prediction
    fingerprint = profile['fingerprint']
    if marker is not None and marker != fingerprint:
        raise ValueError('Prediction score calibration profile changed')
    has_audio = any(view['modality'] == 'A' for view in views)
    if marker is not None and prediction.get('score_calibration_has_audio') != has_audio:
        raise ValueError('Prediction score calibration observation modalities changed')
    if prediction['status'] != 'ok' or not prediction.get('probability_valid', True):
        # Invalid output placeholders are not native probabilities or paired targets.
        return prediction | {'raw_p': None, 'score_calibration': fingerprint,
                             'score_calibration_has_audio': has_audio}
    from .support.score_calibration import probability
    raw = prediction['p'] if marker is None else prediction.get('raw_p')
    if raw is None:
        raise ValueError('Calibrated prediction is missing its native probability')
    calibrated = probability(raw, has_audio, profile)
    if marker is not None:
        if not np.isclose(prediction['p'], calibrated, rtol=0., atol=1e-12):
            raise ValueError('Prediction was transformed more than once or lost its native score')
        return prediction
    if prediction.get('raw_p', raw) != raw:
        raise ValueError('Native prediction has an ambiguous probability scale')
    return prediction | {'raw_p': raw, 'p': calibrated, 'score_calibration': fingerprint,
                         'score_calibration_has_audio': has_audio}


def validate_gain_profile(example, settings, score_calibration=None):
    profile = gain_loss_settings({'gain_loss': example.get('gain_loss', {})})
    if profile != settings:
        raise ValueError('Teacher gain-loss profile changed; regenerate paired targets')
    score_profile = _score_profile(score_calibration)
    if _score_profile(example.get('score_calibration')) != score_profile:
        raise ValueError('Teacher score calibration changed; rederive its paired targets')
    if ((settings['mode'] == 'source_cut_centered_squared' or score_profile is not None)
            and example.get('supervised_valid', True)):
        fields = ('before_probabilities', 'after_probabilities', 'label')
        if any(field not in example for field in fields):
            raise ValueError('Source-centered teacher targets require raw probability triplets and label')
        label = example['label']
        if label not in (0, 1):
            raise ValueError('Paired teacher label must be binary')
        before, after = example['before_probabilities'], example['after_probabilities']
        if score_profile is not None:
            from .support.score_calibration import probability
            for side in ('before', 'after'):
                flag = example.get(side+'_has_audio')
                if not isinstance(flag, bool):
                    raise ValueError('Calibrated teacher requires before/after Audio presence flags')
                expected = np.asarray(probability(example[side+'_probabilities'], flag, score_profile), dtype=float)
                supplied = np.asarray(example.get('calibrated_'+side+'_probabilities'), dtype=float)
                if expected.shape != (3,) or supplied.shape != (3,) or not np.allclose(supplied, expected, rtol=1e-7, atol=1e-8):
                    raise ValueError('Teacher calibrated triplets do not match their native scores/profile')
            before = example['calibrated_before_probabilities']
            after = example['calibrated_after_probabilities']
            if not np.isclose(example['state'][0], before[0], rtol=1e-7, atol=1e-8):
                raise ValueError('Controller state probability is on a different score scale')
        gain, variance, response = paired_targets(before, after, label, settings)
        if (not np.allclose([example['gain'], example['variance']], [gain, variance], rtol=1e-7, atol=1e-8)
                or not np.allclose(example['response'], response, rtol=1e-7, atol=1e-8)):
            raise ValueError('Cached teacher targets do not match their gain-loss profile')


COST_BASIS = 'views_audio_seconds_v1'


def cost_features(views, basis=COST_BASIS):
    if basis not in (COST_BASIS, 'legacy_total_duration'):
        raise ValueError('Unsupported acquisition cost basis')
    seconds = sum(v['stop']-v['start'] for v in views
                  if basis == 'legacy_total_duration' or v['modality'] == 'A')
    # Text/feature observations pay through view count, never a fictitious audio duration.
    return [1., len(views), seconds]


def cost_from_features(features, model, basis=None):
    if isinstance(model, dict):
        if model.get('basis') != COST_BASIS or basis != COST_BASIS:
            raise ValueError('Cost basis changed: recompute teacher timings and features')
        coefficients = model['coefficients']
    else:
        if basis not in (None, 'legacy_total_duration'):
            raise ValueError('New audio-only cost features require their fitted cost basis')
        coefficients = model
    return max(1e-6, float(np.dot(features, coefficients)))


def reader_gate_signature(gate):
    return grpo.signature(gate) if gate is not None else None


def reader_allowed(backbone, gate=None):
    if gate is not None and not isinstance(gate.get('accept'), bool):
        raise ValueError('Reader gate must contain a source-development accept boolean')
    return getattr(backbone, 'reader', None) is not None and (gate is None or gate['accept'])


def reader_state_evidence(backbone, views, gate=None):
    """Read Controller state independently without changing its screening prediction."""
    signature = reader_gate_signature(gate)
    if not reader_allowed(backbone, gate):
        decision = 'source_gate_rejected' if getattr(backbone, 'reader', None) is not None else 'reader_unavailable'
        return [], {'state_reader_used': False, 'state_reader_decision': decision,
                    'state_reader_seconds': 0., 'reader_gate_signature': signature}
    started = time.perf_counter()
    try:
        evidence = backbone.reader.read(views)
    except (ObservationError, FileNotFoundError) as exc:
        return [], {'state_reader_used': False, 'state_reader_decision': 'reader_input_failure_fallback',
                    'state_reader_seconds': time.perf_counter()-started,
                    'state_reader_error': str(exc), 'reader_gate_signature': signature}
    return evidence, {'state_reader_used': bool(evidence),
                      'state_reader_decision': 'accepted' if evidence else 'no_reliable_records',
                      'state_reader_seconds': time.perf_counter()-started,
                      'reader_gate_signature': signature}


def controller_state_evidence(backbone, views, gate=None, prediction=None, state_evidence='gated'):
    """Read only acquired views for the Controller; screening's gate is independent."""
    mode = controller_state_evidence_mode({'state_evidence': state_evidence})
    signature = reader_gate_signature(gate)
    if mode == 'gated':
        if prediction is None:
            return reader_state_evidence(backbone, views, gate)
        evidence = prediction.get('evidence') or []
        return evidence, {'state_reader_used': bool(evidence),
                          'state_reader_decision': prediction.get('reader_decision', 'gated_prediction'),
                          'state_reader_seconds': 0., 'reader_gate_signature': signature}
    reader = getattr(backbone, 'reader', None)
    if reader is None:
        raise ValueError('Observed Reader state evidence requires a restored frozen Reader')
    observed = [view['id'] for view in canonical(views)]
    if prediction is not None and prediction.get('reader_used', False):
        evidence = prediction.get('evidence') or []
        seconds, decision = 0., 'screening_records_reused'
    elif prediction is not None and prediction.get('reader_decision') == 'no_reliable_records':
        evidence, seconds, decision = [], 0., 'no_reliable_records'
    else:
        started = time.perf_counter()
        try:
            with torch.no_grad():
                evidence = reader.read(views)
        except (ObservationError, FileNotFoundError) as exc:
            return [], {'state_reader_used': False, 'state_reader_decision': 'reader_input_failure_fallback',
                        'state_reader_seconds': time.perf_counter()-started, 'state_reader_error': str(exc),
                        'state_evidence': mode, 'state_reader_observed_view_ids': observed,
                        'state_reader_fingerprint': getattr(reader, 'fingerprint', None),
                        'reader_gate_signature': signature}
        seconds = time.perf_counter()-started
        decision = 'observed_reader' if evidence else 'no_reliable_records'
    if any(record.get('source_id') not in observed for record in evidence):
        raise ValueError('Controller Reader records reference unobserved views')
    return evidence, {'state_reader_used': bool(evidence), 'state_reader_decision': decision,
                      'state_reader_seconds': seconds, 'state_evidence': mode,
                      'state_reader_observed_view_ids': observed,
                      'state_reader_fingerprint': getattr(reader, 'fingerprint', None),
                      'reader_gate_signature': signature}


def score_with_reader_gate(backbone, views, reader_gate=None, renderer=0, repeat_id=None,
                           plain=None, evidence=None, capture_controller_state=False, score_calibration=None):
    """Use the source-selected Reader branch, falling back to a genuine plain score."""
    if capture_controller_state and (renderer != 0 or repeat_id is not None):
        raise RuntimeError('Controller context must use the canonical screening renderer')
    signature = reader_gate_signature(reader_gate)
    plain_score = getattr(backbone, 'score_unaugmented', backbone.score)
    capture_options = {'capture_controller_state': True} if capture_controller_state else {}
    def verified_prediction(prediction):
        if capture_controller_state and prediction['status'] == 'ok' and prediction.get('probability_valid', True):
            controller_context_rows(views, prediction.get('controller_context'))
        return calibrate_prediction(prediction, views, score_calibration)
    def fallback_score():
        prediction = plain if plain is not None else plain_score(views, renderer, repeat_id, **capture_options)
        return verified_prediction(prediction)
    allowed = reader_allowed(backbone, reader_gate)
    if allowed:
        reader_seconds, read_failure = 0., None
        if evidence is None and hasattr(backbone.reader, 'read'):
            read_start = time.perf_counter()
            try:
                evidence = backbone.reader.read(views)
            except (ObservationError, FileNotFoundError) as exc:
                read_failure = str(exc)
            reader_seconds = time.perf_counter()-read_start
        if read_failure is not None or evidence == []:
            fallback = fallback_score()
            result = fallback | {'seconds': reader_seconds+(fallback['seconds'] if plain is None else 0.),
                                 'reader_used': False, 'reader_gate_signature': signature,
                                 'reader_decision': ('reader_input_failure_fallback' if read_failure is not None
                                                     else 'no_reliable_records')}
            if read_failure is not None:
                result['reader_failure'] = {'status': 'input_failure', 'error': read_failure}
            return result
        enhanced = verified_prediction(backbone.score(views, renderer, repeat_id, evidence=evidence,
                                                     **capture_options))
        enhanced = enhanced | {'seconds': enhanced['seconds']+reader_seconds}
        if enhanced['status'] == 'ok' and enhanced.get('probability_valid', True):
            return enhanced | {'reader_used': True, 'reader_decision': 'accepted',
                               'reader_gate_signature': signature}
        # Do not replace a failed enhanced answer with its neutral placeholder p.
        fallback = fallback_score()
        seconds = enhanced['seconds'] + (fallback['seconds'] if plain is None else 0.)
        return fallback | {'seconds': seconds, 'reader_used': False,
                           'reader_decision': 'enhanced_failure_fallback',
                           'reader_failure': {'status': enhanced['status'], 'error': enhanced.get('error')},
                           'reader_gate_signature': signature}
    fallback = fallback_score()
    return fallback | {'reader_used': False,
                       'reader_decision': ('source_gate_rejected' if getattr(backbone, 'reader', None) is not None
                                           else 'reader_unavailable'),
                       'reader_gate_signature': signature}


def teacher_sampling_settings(config):
    settings = config.get('teacher_sampling', {})
    if not isinstance(settings, dict):
        raise ValueError('Teacher sampling must be a configuration dictionary')
    mode = settings.get('mode', 'modality_uniform')
    if mode not in ('modality_uniform', 'aligned_audio'):
        raise ValueError('Unsupported teacher action sampling mode')
    return {'mode': mode}


def aligned_audio_candidates(record, state, candidates):
    """Shared-source audio order used by teacher sampling and source validation."""
    state_ids = tuple(v['id'] for v in canonical(state))
    events = {v['event'] for v in state}
    eligible = [v for v in candidates if v['modality'] == 'A' and v['event'] in events
                and v['id'] not in state_ids and v.get('available', True)]
    return sorted(eligible, key=lambda v: (digest((record['family'], record['id'], state_ids, v['id'])), v['id']))


def teacher_action_indices(record, state, remaining, count_actions, rng, sampling):
    picks = []
    for modality in sorted({v['modality'] for v in remaining}):
        if len(picks) < count_actions:
            eligible = [j for j, v in enumerate(remaining) if v['modality'] == modality]
            picked = int(rng.choice(eligible))
            if modality == 'A' and sampling['mode'] == 'aligned_audio':
                aligned = aligned_audio_candidates(record, state, [remaining[j] for j in eligible])
                if aligned:
                    picked = next(j for j in eligible if remaining[j]['id'] == aligned[0]['id'])
            # Consume the original modality draw even when replacing its A slot.
            picks.append(picked)
    rest = [j for j in range(len(remaining)) if j not in picks]
    if len(picks) < count_actions:
        picks.extend(rng.choice(rest, count_actions-len(picks), replace=False).tolist())
    return picks


def teacher_pairs(backbone, records, track, config):
    gain_loss = gain_loss_settings(config)
    score_profile = _score_profile(config.get('score_calibration'))
    calibration_options = {'score_calibration': score_profile} if score_profile is not None else {}
    sampling = teacher_sampling_settings(config)
    from .support.controller_rollouts import teacher_rollout_settings, aligned_audio_rollout_plan
    rollout = teacher_rollout_settings(config)
    rng = np.random.default_rng(config['seed'])
    examples, timings = [], []
    seen_pairs, seen_states = set(), set()
    reported_failures = set()
    max_pairs = config.get('max_teacher_pairs')
    max_states = config.get('max_teacher_states')
    budget = float(config.get('budget', 8))
    reader_gate = config.get('reader_gate')
    use_reader = reader_allowed(backbone, reader_gate)
    state_evidence_mode = controller_state_evidence_mode(config)
    candidate_mode = candidate_conditioning_mode(config)
    use_state_reader = use_reader or state_evidence_mode == 'observed_reader'
    state_representation, seed_anchor = controller_inputs(config)
    capture_context = state_representation == 'frozen_discriminator_context'
    capture_options = {'capture_controller_state': True} if capture_context else {}
    plain_anchor = seed_anchor == 'plain_until_acquisition'
    def probability_valid(prediction):
        return prediction['status'] == 'ok' and prediction.get('probability_valid', True)
    def generated_output_failure(prediction):
        return (prediction.get('output_generated', False)
                and prediction.get('failure_stage') == 'screening_output')
    def report_invalid(rec, stage, view, predictions):
        for renderer, prediction in enumerate(predictions):
            if prediction['status'] == 'ok':
                continue
            failure = {'rec': rec['id'], 'split': rec.get('split'), 'stage': stage,
                       'view': view, 'render': renderer, 'status': prediction['status'],
                       'error': prediction.get('error'), 'answer': str(prediction.get('answer', ''))[:200]}
            key = json.dumps(failure, ensure_ascii=False, sort_keys=True)
            if key not in reported_failures:
                reported_failures.add(key)
                print('teacher_invalid: '+key, flush=True)
    if use_state_reader:
        from .support.reader_training import index_reader_positions
        index_reader_positions(records, backbone.reader.config)
    for rec in records:
        try:
            initial = seed_views(rec, track)
        except ObservationError:
            continue
        options = [v for v in pool(rec, track) if v not in initial]
        rollout_plans = None
        if rollout['mode'] == 'seed_aligned_audio':
            options, rollout_plans = aligned_audio_rollout_plan(rec, track, initial, options)
        plain_score = getattr(backbone, 'score_unaugmented', backbone.score)
        reference = (calibrate_prediction(plain_score(initial, **capture_options), initial, score_profile)
                     if plain_anchor or (getattr(backbone, 'reader', None) is not None and hasattr(backbone, 'score_unaugmented')) else None)
        if reference is not None and not probability_valid(reference):
            report_invalid(rec, 'seed_plain', [v['id'] for v in initial], [reference])
            continue
        cache = {}
        selected_cache = {}
        initial_key = tuple(v['id'] for v in canonical(initial))
        state_evidence_cache, state_reader_metadata, state_timed, state_charged = {}, {}, set(), set()
        orders = range(len(rollout_plans)) if rollout_plans is not None else ('chronological', 'uniform', 'random')
        for order in orders:
            ordered = list(options)
            if rollout_plans is not None:
                ordered, prefix = rollout_plans[order]
            elif order == 'random':
                rng.shuffle(ordered)
            elif order == 'uniform':
                indices = np.linspace(0, len(options)-1, min(8, len(options)), dtype=int) if options else []
                selected = [options[i] for i in indices]
                ordered = selected + [v for v in options if v not in selected]
            for count in ([prefix] if rollout_plans is not None else config['prefix_lengths']):
                state = canonical(initial + ordered[:count])
                remaining = [v for v in options if v not in state]
                if not remaining:
                    continue
                state_id = (rec['id'], tuple(v['id'] for v in state))
                if state_id not in seen_states:
                    if max_states is not None and len(seen_states) >= int(max_states):
                        break
                    seen_states.add(state_id)
                # The source seed anchor remains observed on every audio-only
                # branch. Holding it fixed also reuses each matched C_r score
                # when the same subset is reached through another transition.
                anchor = initial[0]['id'] if rollout_plans is not None else state[0]['id']
                def evaluate(views, renderer):
                    key = (tuple(v['id'] for v in canonical(views)), renderer, anchor)
                    score_repeat_id = None if capture_context and renderer == 0 else anchor
                    if key not in cache:
                        # First canonical score includes rereading cost. All C_r share its records.
                        state_key = key[0]
                        if plain_anchor and state_key == initial_key:
                            if state_key not in state_evidence_cache:
                                evidence, read_meta = controller_state_evidence(
                                    backbone, views, reader_gate, state_evidence=state_evidence_mode)
                                state_evidence_cache[state_key] = evidence
                                state_reader_metadata[state_key] = read_meta
                            prediction = (reference if renderer == 0 else
                                          calibrate_prediction(plain_score(views, renderer, anchor), views, score_profile))
                            cache[key] = prediction | {'reader_used': False,
                                                       'reader_decision': 'plain_until_acquisition',
                                                       **state_reader_metadata[state_key]}
                            if renderer == 0 and prediction['status'] == 'ok' and state_evidence_mode == 'gated':
                                encoding_start = time.perf_counter()
                                encode_state(views, state_evidence_cache[state_key], backbone, state_representation,
                                             controller_context=prediction.get('controller_context'))
                                seconds = (prediction['seconds']+state_reader_metadata[state_key]['state_reader_seconds']
                                           +time.perf_counter()-encoding_start)
                                timings.append(cost_features(views)[1:]+[seconds, COST_BASIS])
                        elif use_reader:
                            if state_key not in selected_cache:
                                canonical_score = score_with_reader_gate(
                                    backbone, views, reader_gate, 0, None if capture_context else anchor,
                                    plain=reference if state_key == tuple(v['id'] for v in initial) else None,
                                    **capture_options, **calibration_options)
                                selected_cache[state_key] = canonical_score
                                cache[(state_key, 0, anchor)] = canonical_score
                                if canonical_score['status'] == 'ok' and state_evidence_mode == 'gated':
                                    encoding_start = time.perf_counter()
                                    encode_state(views, canonical_score.get('evidence'), backbone, state_representation,
                                                 controller_context=canonical_score.get('controller_context'))
                                    seconds = canonical_score['seconds']+time.perf_counter()-encoding_start
                                    timings.append(cost_features(views)[1:]+[seconds, COST_BASIS])
                            if capture_context and renderer == 0:
                                # The same acquired state can recur under a different
                                # parent anchor; reuse its complete canonical prefill.
                                cache[key] = selected_cache[state_key]
                            if key not in cache:
                                selected = selected_cache[state_key]
                                if selected['reader_used']:
                                    cache[key] = score_with_reader_gate(
                                        backbone, views, reader_gate, renderer, score_repeat_id,
                                        evidence=selected.get('evidence', []), **calibration_options)
                                else:
                                    # Matched renderers retain the canonical branch and records.
                                    cache[key] = calibrate_prediction(plain_score(views, renderer, score_repeat_id), views, score_profile) | {
                                        'reader_used': False, 'reader_decision': selected['reader_decision'],
                                        'reader_gate_signature': reader_gate_signature(reader_gate)}
                        else:
                            cache[key] = score_with_reader_gate(
                                backbone, views, reader_gate, renderer, score_repeat_id,
                                plain=reference if renderer == 0 and state_key == tuple(v['id'] for v in initial) else None,
                                **(capture_options if renderer == 0 else {}), **calibration_options)
                            if renderer == 0 and cache[key]['status'] == 'ok' and state_evidence_mode == 'gated':
                                encoding_start = time.perf_counter()
                                encode_state(views, cache[key].get('evidence'), backbone, state_representation,
                                             controller_context=cache[key].get('controller_context'))
                                seconds = cache[key]['seconds']+time.perf_counter()-encoding_start
                                timings.append(cost_features(views)[1:]+[seconds, COST_BASIS])
                    if renderer == 0 and state_evidence_mode == 'observed_reader' and cache[key]['status'] == 'ok':
                        state_key = key[0]
                        if state_key not in state_evidence_cache:
                            state_records, read_meta = controller_state_evidence(
                                backbone, views, reader_gate, prediction=cache[key],
                                state_evidence=state_evidence_mode)
                            state_evidence_cache[state_key], state_reader_metadata[state_key] = state_records, read_meta
                        if key not in state_charged:
                            read_meta = state_reader_metadata[state_key]
                            cache[key] = cache[key] | read_meta | {
                                'seconds': cache[key]['seconds']+read_meta['state_reader_seconds']}
                            state_charged.add(key)
                        if state_key not in state_timed:
                            encoding_start = time.perf_counter()
                            encode_state(views, state_evidence_cache[state_key], backbone, state_representation,
                                         controller_context=cache[key].get('controller_context'))
                            seconds = cache[key]['seconds']+time.perf_counter()-encoding_start
                            timings.append(cost_features(views)[1:]+[seconds, COST_BASIS])
                            state_timed.add(state_key)
                    return cache[key]
                before = [evaluate(state, r) for r in range(3)]
                if (not probability_valid(before[0])
                        or any(not probability_valid(x) and not generated_output_failure(x) for x in before)):
                    report_invalid(rec, 'before', [v['id'] for v in state], before)
                    continue
                screening_config = getattr(backbone, 'screening', None)
                before_valid = all(probability_valid(x) for x in before)
                before_reward = None
                if screening_config:
                    rewards = [screening_record_reward(x['answer'], rec, screening_config) for x in before]
                    before_reward = rewards[0] if before_valid else min(rewards, key=lambda x: x['total'])
                seed_seconds = max(1e-6, (reference or evaluate(initial, 0))['seconds'] + rec.get('index_seconds', 0.))
                initial_prediction = evaluate(initial, 0)
                initial_extra = (state_reader_metadata[initial_key]['state_reader_seconds'] if plain_anchor
                                 else initial_prediction['seconds'] if reference and use_reader else 0.)
                if state_evidence_mode == 'observed_reader' and not plain_anchor and not (reference and use_reader):
                    initial_extra = state_reader_metadata[initial_key]['state_reader_seconds']
                elapsed = seed_seconds + initial_extra + sum(evaluate(canonical(initial+ordered[:k]), 0)['seconds']
                                             for k in range(1, min(count, len(ordered))+1))
                budget_fraction = max(0., 1-elapsed/(budget*seed_seconds))
                count_actions = (len(remaining) if rollout_plans is not None else
                                 min(int(config['actions_per_state']), len(remaining)))
                # Include each available modality before filling the sample. This
                # exercises both interfaces even in a two-action bounded run.
                picks = (list(range(len(remaining))) if rollout_plans is not None else
                         teacher_action_indices(rec, state, remaining, count_actions, rng, sampling))
                evidence = state_evidence_cache.get(tuple(v['id'] for v in state), before[0].get('evidence'))
                x = encode_state(state, evidence, backbone, state_representation,
                                 controller_context=before[0].get('controller_context'))
                for j in picks:
                    action = remaining[j]
                    pair_id = (state_id, action['id'])
                    if pair_id in seen_pairs:
                        continue
                    after = [evaluate(canonical(state+[action]), r) for r in range(3)]
                    after_valid = all(probability_valid(x) for x in after)
                    if not after_valid:
                        report_invalid(rec, 'after', action['id'], after)
                    if (any(not probability_valid(x) and not generated_output_failure(x) for x in after)
                            or (not after_valid and not screening_config)):
                        # Input/context/runtime failures are not model-generated format mistakes.
                        continue
                    supervised_valid = before_valid and after_valid
                    examples.append({'participant': rec['family']+':'+rec['participant'],
                                     'state_id': state_id, 'action_id': action['id'], 'x': x,
                                     'state': np.array([before[0]['p'], 1., budget_fraction], dtype=np.float32),
                                     'candidate': metadata(action, state),
                                     'supervised_valid': supervised_valid,
                                     'cost_features': cost_features(state+[action]), 'cost_basis': COST_BASIS,
                                     'seed_seconds': seed_seconds,
                                     'elapsed_seconds': elapsed,
                                     'state_representation': state_representation,
                                     'state_representation_version': STATE_REPRESENTATION_VERSION,
                                     'seed_prediction_anchor': seed_anchor,
                                     'state_evidence': state_evidence_mode,
                                     'candidate_conditioning': candidate_mode,
                                     'gain_loss': gain_loss,
                                     'teacher_sampling': sampling,
                                     'teacher_rollout': rollout,
                                     'schema_valid': after[0].get('schema_valid'),
                                     'safety_valid': after[0].get('safety_valid')})
                    if capture_context:
                        examples[-1]['controller_context'] = copy.deepcopy(before[0]['controller_context'])
                    if candidate_mode != 'none':
                        examples[-1]['candidate_mask'] = encode_candidate_mask(
                            state, evidence, action, state_representation, candidate_mode)
                    if supervised_valid:
                        before_probabilities = [b.get('raw_p', b['p']) for b in before]
                        after_probabilities = [b.get('raw_p', b['p']) for b in after]
                        calibrated_before, calibrated_after = [b['p'] for b in before], [b['p'] for b in after]
                        gain, variance, response = paired_targets(
                            calibrated_before, calibrated_after, rec['label'], gain_loss)
                        examples[-1].update(gain=gain, variance=variance, response=response,
                                           before_probabilities=before_probabilities,
                                           after_probabilities=after_probabilities, label=rec['label'])
                        if score_profile is not None:
                            examples[-1].update(calibrated_before_probabilities=calibrated_before,
                                                calibrated_after_probabilities=calibrated_after)
                    else:
                        examples[-1]['output_failures'] = [
                            {'renderer': r, 'status': p['status'], 'error': p.get('error')}
                            for r, p in enumerate(after) if not probability_valid(p)]
                    examples[-1]['reader_fingerprint'] = getattr(getattr(backbone, 'reader', None), 'fingerprint', None)
                    examples[-1]['reader_gate_signature'] = reader_gate_signature(reader_gate)
                    examples[-1]['reader_used'] = before[0].get('reader_used', False)
                    examples[-1]['state_reader_used'] = bool(evidence)
                    if state_evidence_mode == 'observed_reader' or candidate_mode != 'none':
                        examples[-1].update(state_reader_records=evidence,
                                            state_reader_seconds=state_reader_metadata.get(state_id[1], {}).get('state_reader_seconds', 0.))
                    if score_profile is not None:
                        examples[-1].update(score_calibration=score_profile,
                                            before_has_audio=any(v['modality'] == 'A' for v in state),
                                            after_has_audio=any(v['modality'] == 'A' for v in state+[action]))
                    if before_reward is not None:
                        rewards = [screening_record_reward(p['answer'], rec, screening_config) for p in after]
                        # For malformed presentation variants, keep the actual worst
                        # generated reward without inventing a Brier probability.
                        after_reward = rewards[0] if after_valid else min(rewards, key=lambda x: x['total'])
                        # Baseline screening rewards augment only GRPO, never the
                        # paired Brier gain/sensitivity/response supervision.
                        examples[-1]['screening_reward_delta'] = {
                            key: float(after_reward[key]-before_reward[key])
                            for key in ('accuracy', 'structure', 'safety', 'total')}
                    seen_pairs.add(pair_id)
                    detail = f"gain={examples[-1]['gain']:.6f}" if supervised_valid else 'reward_only_output_failure'
                    print(f"teacher_pair: {rec['id']} {action['modality']} {action['id']} {detail}", flush=True)
                    if max_pairs is not None and len(examples) >= int(max_pairs):
                        print(f'teacher_pairs: {len(examples)} pairs, {len(seen_states)} states (configured cap)', flush=True)
                        return examples, timings
        if max_states is not None and len(seen_states) >= int(max_states):
            break
    if not examples:
        raise ValueError('No eligible teacher actions; inspect source eligibility and interfaces')
    return examples, timings


def teacher_augmentation_settings(config):
    settings = config.get('teacher_augmentation', {})
    if not isinstance(settings, dict):
        raise ValueError('Teacher augmentation must be a configuration dictionary')
    mode = settings.get('mode', 'disabled')
    if mode == 'disabled':
        return {'mode': mode}
    if mode != 'aligned_audio' or settings.get('actions_per_state', 1) != 1:
        raise ValueError('Aligned-audio augmentation allows one new action per original state')
    return {'mode': mode, 'actions_per_state': 1}


def augment_teacher_pairs(backbone, records, track, config, existing_pairs, directory=None, resume=False):
    """Add one missing shared audio action per cached state, never score before.

    Returns added pairs and new timings only. The experiments caller exclusively
    owns record caches/resume and merges the untouched existing target prefix.
    """
    if directory is not None or resume:
        raise ValueError('Teacher augmentation record cache/resume belongs to the experiments caller')
    settings = teacher_augmentation_settings(config)
    stats = {'mode': settings['mode'], 'states': 0, 'added': 0, 'skipped': {}}
    backbone.teacher_augmentation_stats = stats
    if settings['mode'] == 'disabled':
        return [], []
    if config.get('grpo', {}).get('enabled', True):
        raise ValueError('Cached before answers cannot supply complete GRPO rewards; augmentation requires Controller GRPO disabled')
    gain_loss = gain_loss_settings(config)
    gate = config.get('reader_gate')
    score_profile = _score_profile(config.get('score_calibration'))
    calibration_options = {'score_calibration': score_profile} if score_profile is not None else {}
    capture_context = config.get('state_representation') == 'frozen_discriminator_context'
    capture_options = {'capture_controller_state': True} if capture_context else {}
    use_reader = reader_allowed(backbone, gate)
    state_evidence_mode = controller_state_evidence_mode(config)
    candidate_mode = candidate_conditioning_mode(config)
    use_state_reader = use_reader or state_evidence_mode == 'observed_reader'
    plain_score = getattr(backbone, 'score_unaugmented', backbone.score)
    added, timings = [], []
    def skip(reason):
        stats['skipped'][reason] = stats['skipped'].get(reason, 0)+1
    for rec in records:
        if rec.get('split') not in ('train', 'dev'):
            raise ValueError('Teacher augmentation is restricted to source train/development records')
        participant = rec['family']+':'+rec['participant']
        original = [p for p in existing_pairs if p['participant'] == participant
                    and str(p['state_id'][0]) == str(rec['id'])]
        grouped = {}
        for pair in original:
            grouped.setdefault(tuple(pair['state_id'][1]), []).append(pair)
        views = {v['id']: v for v in pool(rec, track)}
        for state_ids, actions in grouped.items():
            stats['states'] += 1
            if any(identity not in views for identity in state_ids):
                raise ValueError('Cached teacher state references missing source observations')
            state = canonical([views[identity] for identity in state_ids])
            valid = [p for p in actions if p.get('supervised_valid', True) and 'before_probabilities' in p]
            if not valid:
                skip('missing_valid_raw_before')
                continue
            template = valid[0]
            validate_gain_profile(template, gain_loss, score_profile)
            if gain_loss_settings({'gain_loss': template.get('gain_loss', {})}) != gain_loss:
                raise ValueError('Aligned-audio targets must preserve the original gain-loss profile')
            if controller_inputs(template) != controller_inputs(config):
                raise ValueError('Aligned-audio targets must preserve original Controller input/anchor')
            if controller_state_evidence_mode(template) != state_evidence_mode:
                raise ValueError('Aligned-audio targets must preserve original Controller state evidence')
            if candidate_conditioning_mode(template) != candidate_mode:
                raise ValueError('Aligned-audio targets must preserve original candidate conditioning')
            if template.get('reader_gate_signature') != reader_gate_signature(gate):
                raise ValueError('Aligned-audio targets must preserve the source Reader gate')
            if template['label'] != rec['label']:
                raise ValueError('Cached teacher source label changed')
            before = np.asarray(template['before_probabilities'], dtype=float)
            if before.shape != (3,) or not np.isfinite(before).all():
                raise ValueError('Aligned-audio augmentation requires three valid cached before probabilities')
            covered = {p['action_id'] for p in actions}
            candidates = [v for v in views.values() if v['modality'] == 'A' and v['id'] not in state_ids
                          and v['id'] not in covered and metadata(v, state)[6] == 1.]
            if float(template['state'][2]) <= 0:
                skip('no_remaining_state_budget')
                continue
            eligible = []
            for action in candidates:
                combined = canonical(state+[action])
                if use_state_reader:
                    reader = backbone.reader
                    if reader.candidate_capacity_exceeded(state, action):
                        continue
                if config.get('cost_model') is not None:
                    predicted = predicted_cost(combined, config['cost_model'])
                    remaining = float(template['state'][2])*float(config.get('budget', 8))*float(template['seed_seconds'])
                    if predicted > remaining:
                        continue
                eligible.append(action)
            if not eligible:
                skip('no_missing_eligible_shared_audio')
                continue
            # Deterministic source identity hash; labels and predicted gains do not select the action.
            action = aligned_audio_candidates(rec, state, eligible)[0]
            combined, anchor = canonical(state+[action]), state[0]['id']
            canonical_score = score_with_reader_gate(backbone, combined, gate, 0,
                                                     None if capture_context else anchor,
                                                     **capture_options, **calibration_options)
            if canonical_score['status'] == 'ok' and canonical_score.get('probability_valid', True):
                state_records, read_meta = controller_state_evidence(
                    backbone, combined, gate, prediction=canonical_score, state_evidence=state_evidence_mode)
                started = time.perf_counter()
                encode_state(combined, state_records, backbone, config.get('state_representation', 'hash'),
                             controller_context=canonical_score.get('controller_context'))
                seconds = canonical_score['seconds']+read_meta['state_reader_seconds']+time.perf_counter()-started
                timings.append(cost_features(combined)[1:]+[seconds, COST_BASIS])
            after = [canonical_score]
            for renderer in (1, 2):
                if canonical_score.get('reader_used', False):
                    prediction = score_with_reader_gate(backbone, combined, gate, renderer, anchor,
                                                        evidence=canonical_score.get('evidence', []), **calibration_options)
                else:
                    prediction = calibrate_prediction(plain_score(combined, renderer, anchor), combined, score_profile)
                after.append(prediction)
            if any(p['status'] != 'ok' or not p.get('probability_valid', True) for p in after):
                skip('invalid_after_probability')
                print('teacher_augmentation_invalid: '+json.dumps({
                    'rec': rec['id'], 'action': action['id'], 'outputs': [
                        {'renderer': r, 'status': p['status'], 'error': p.get('error'),
                         'answer': str(p.get('answer', ''))[:200]} for r, p in enumerate(after)
                        if p['status'] != 'ok' or not p.get('probability_valid', True)]}), flush=True)
                continue
            raw_after = [p.get('raw_p', p['p']) for p in after]
            calibrated_before = template['calibrated_before_probabilities'] if score_profile is not None else before
            calibrated_after = [p['p'] for p in after]
            gain, variance, response = paired_targets(calibrated_before, calibrated_after, rec['label'], gain_loss)
            pair = dict(template)
            pair.pop('screening_reward_delta', None)  # Cached before answers cannot justify new reward deltas.
            pair.update(action_id=action['id'], candidate=metadata(action, state),
                        cost_features=cost_features(combined), cost_basis=COST_BASIS,
                        gain=gain, variance=variance, response=response, after_probabilities=raw_after,
                        schema_valid=after[0].get('schema_valid'), safety_valid=after[0].get('safety_valid'),
                        teacher_augmentation=settings)
            if candidate_mode != 'none':
                original_evidence = template.get('state_reader_records', template.get('reader_evidence', []))
                pair['candidate_mask'] = encode_candidate_mask(
                    state, original_evidence, action, config.get('state_representation', 'hash'), candidate_mode)
            if score_profile is not None:
                pair.update(calibrated_after_probabilities=calibrated_after,
                            after_has_audio=any(v['modality'] == 'A' for v in combined))
            added.append(pair)
            stats['added'] += 1
            print('teacher_augmentation: '+json.dumps({'rec': rec['id'], 'action': action['id'],
                                                      'gain': gain, 'shared': 1}), flush=True)
    print('teacher_augmentation_summary: '+json.dumps(stats), flush=True)
    return added, timings


def sensitivity_from_log_variance(q):
    """Deployment's standard deviation with a finite zero-variance derivative."""
    variance = (q.clamp(-30, 30).exp()-1e-6).clamp_min(0.)
    positive = variance > 0
    # The inactive sqrt branch sees one, avoiding sqrt(0)'s infinite backward.
    safe_variance = torch.where(positive, variance, torch.ones_like(variance))
    return torch.where(positive, safe_variance.sqrt(), torch.zeros_like(variance))


def candidate_options(example):
    return ({'candidate_mask': example['candidate_mask']}
            if candidate_conditioning_mode(example) != 'none' else {})


def loss(model, example, alpha, beta, sensitivity_loss='log_variance_mse'):
    x, z = example['x'].copy(), example['candidate'].copy()
    normalized = getattr(model, 'target_normalization', 'none') != 'none'
    options = dict(candidate_options(example), **({'normalized': True} if normalized else {}))
    g, q, n = model(torch.tensor(x), torch.tensor(example['state']), torch.tensor(z), **options)
    scale = float(model.target_scale) if normalized else 1.
    gain_target = example['gain']/scale
    variance_target = example['variance']/(scale*scale)
    if sensitivity_loss == 'std_mse':
        sensitivity = sensitivity_from_log_variance(q)
        auxiliary = alpha*(sensitivity-np.sqrt(max(0., variance_target)))**2
    elif sensitivity_loss == 'log_variance_mse':
        auxiliary = alpha*(q-np.log(variance_target+1e-6))**2
    else:
        raise ValueError('Unsupported Controller sensitivity loss')
    main = (g-gain_target)**2 + auxiliary
    response_target = torch.tensor(example['response'], dtype=torch.float32)
    if normalized:
        response_target = response_target/scale
    null = beta*((n-response_target)**2).mean()
    return main, null, g*scale if normalized else g


def decision_config(config):
    settings = {'enabled': False, 'mode': 'categorical', 'weight': .1,
                'temperature': .1, 'class_balance': 'none', 'candidate_ranking_weight': 0.}
    supplied = config.get('decision_loss', {})
    if not isinstance(supplied, dict):
        raise ValueError('Controller decision_loss must be a configuration dictionary')
    settings.update(supplied)
    if (not isinstance(settings['enabled'], bool)
            or not np.isfinite([settings['weight'], settings['temperature'],
                                settings['candidate_ranking_weight']]).all()
            or settings['weight'] < 0 or settings['temperature'] <= 0
            or settings['candidate_ranking_weight'] < 0
            or settings['mode'] not in ('categorical', 'binary_stop')
            or settings['class_balance'] not in ('none', 'stop_acquire')):
        raise ValueError('Invalid Controller decision-loss configuration')
    if settings['enabled'] and settings['weight'] == 0:
        raise ValueError('Enabled Controller decision loss requires a positive weight')
    if settings['candidate_ranking_weight'] and (not settings['enabled'] or settings['mode'] != 'binary_stop'):
        raise ValueError('Candidate ranking requires the binary action-or-STOP objective')
    return settings


def decision_oracle(actions, config, variant='full'):
    """Use only valid paired targets that fit this state's remaining budget."""
    lam = float(config.get('policy_lambda', .01))
    gamma = 0. if variant == 'no_presentation' else float(config.get('policy_gamma', .1))
    eligible = []
    for example in actions:
        if not candidate_action_allowed(example['candidate'], action_filter_mode(config)):
            continue
        if not example.get('supervised_valid', True):
            continue
        cost = float(example.get('cost', 0.))
        if cost > float(example['state'][2])*float(config.get('budget', 8)):
            continue
        if (not np.isfinite([cost, example['gain'], example['variance']]).all()
                or cost < 0 or example['variance'] < 0):
            raise ValueError('Controller decision targets must have finite gain, variance and cost')
        eligible.append(example)
    eligible.sort(key=lambda example: str(example.get('action_id', '')))
    if not eligible:
        return [], None, None
    utilities = [e['gain']-lam*float(e.get('cost', 0.))-gamma*np.sqrt(e['variance']) for e in eligible]
    best = max(range(len(eligible)), key=lambda index: (utilities[index], str(eligible[index].get('action_id', ''))))
    target = best if utilities[best] > 0 else len(eligible)
    return eligible, target, 'acquire' if target < len(eligible) else 'stop'


def training_state_weights(states, mode='uniform'):
    """Match each participant's training objective to initial/later state strata."""
    if mode not in ('uniform', 'initial_later_balanced'):
        raise ValueError('Unsupported Controller training state weighting')
    if not states:
        return {}
    if mode == 'uniform':
        return {state_id: 1./len(states) for state_id in states}
    minimum = {}
    for state_id in states:
        record = str(state_id[0])
        minimum[record] = min(minimum.get(record, len(state_id[1])), len(state_id[1]))
    strata = {True: [], False: []}
    for state_id in states:
        strata[len(state_id[1]) == minimum[str(state_id[0])]].append(state_id)
    present = [state_ids for state_ids in strata.values() if state_ids]
    return {state_id: 1./(len(present)*len(state_ids))
            for state_ids in present for state_id in state_ids}


def fit_target_profile(grouped, config, alpha, beta, variant='full'):
    """TRAIN-only signed-gain RMS under the exact participant/state/action MSE weights."""
    if target_normalization_mode(config) == 'none':
        if config.get('target_profile') is not None:
            raise ValueError('Legacy Controller cannot reuse a normalized target profile')
        return None
    if not grouped:
        raise ValueError('Target normalization requires source training participants')
    supervised_weight = float(config['grpo']['supervised_weight'])
    if not np.isfinite(supervised_weight) or supervised_weight <= 0:
        raise ValueError('Target normalization requires positive paired supervision weight')
    targets, weights = [], []
    square_mass, mass, valid_states = 0., 0., 0
    state_mode = config.get('training_state_weighting', 'uniform')
    for participant, states in grouped.items():
        state_weights = training_state_weights(states, state_mode)
        for state_id, actions in states.items():
            valid = [e for e in actions if e.get('supervised_valid', True)]
            if not valid:
                continue
            valid_states += 1
            weight = supervised_weight*state_weights[state_id]/(len(grouped)*len(valid))
            for example in valid:
                gain, variance = float(example['gain']), float(example['variance'])
                response = np.asarray(example['response'], dtype=float)
                if not np.isfinite([gain, variance]).all() or variance < 0 or not np.isfinite(response).all():
                    raise ValueError('Target normalization requires finite valid paired targets')
                identity = [str(participant), str(state_id[0]), list(map(str, state_id[1])),
                            str(example.get('action_id', ''))]
                targets.append(identity+[gain, variance, response.tolist()])
                weights.append(identity+[weight])
                square_mass += weight*gain*gain
                mass += weight
    if mass <= 0 or not np.isfinite([square_mass, mass]).all():
        raise ValueError('Target normalization requires positive finite TRAIN target mass')
    rms = float(np.sqrt(square_mass/mass))
    # All-zero gains have no gain scale to estimate; retain identity units explicitly.
    scale = rms if rms > 0 else 1.
    loss_settings = {'alpha': float(alpha), 'beta': float(beta), 'variant': variant,
                     'sensitivity_loss': config.get('sensitivity_loss', 'log_variance_mse'),
                     'training_state_weighting': state_mode,
                     'grpo': config['grpo'], 'decision_loss': decision_config(config),
                     'gain_loss': gain_loss_settings(config),
                     'ranking_profile': config.get('ranking_profile')}
    profile = {'version': 1, 'mode': 'source_train_weighted_rms_v1',
               'source': 'source_train_effective_paired_gain_weights', 'scale': scale,
               'centering': False, 'zero_gain_identity': rms == 0.,
               'train_target_signature': _target_signature(targets),
               'weight_signature': _target_signature(weights),
               'loss_signature': _target_signature(loss_settings),
               'counts': {'participants': len(grouped), 'states': valid_states,
                          'valid_pairs': len(targets)},
               'effective_gain_mass': mass, 'weighted_gain_square_mass': square_mass}
    profile['fingerprint'] = _target_signature(profile)
    validate_target_profile(profile)
    if config.get('target_profile') is not None and validate_target_profile(config['target_profile']) != profile:
        raise ValueError('Controller target profile does not match current source TRAIN targets/weights')
    return profile


def decision_training_statistics(grouped, config, settings, variant='full'):
    counts = {'stop': 0, 'acquire': 0}
    mode = config.get('training_state_weighting', 'uniform')
    masses = {'stop': 0., 'acquire': 0.}
    for states in grouped.values():
        state_weights = training_state_weights(states, mode)
        for state_id, actions in states.items():
            _, _, category = decision_oracle(actions, config, variant)
            if category is not None:
                counts[category] += 1
                masses[category] += state_weights[state_id]/len(grouped)
    weights = {'stop': 1., 'acquire': 1.}
    balance = masses if mode == 'initial_later_balanced' else counts
    if settings['class_balance'] == 'stop_acquire' and all(balance.values()):
        total = sum(balance.values())
        weights = {category: total/(2*mass) for category, mass in balance.items()}
    result = {'source': 'training_states_only', 'class_counts': counts,
              'class_weights': weights, 'eligible_states': sum(counts.values())}
    if mode == 'initial_later_balanced':
        result.update(training_state_weighting=mode, effective_class_mass=masses,
                      effective_eligible_mass=sum(masses.values()))
    return result


def decision_choice_losses(model, actions, config, settings, statistics, variant='full'):
    eligible, target, category = decision_oracle(actions, config, variant)
    if not eligible:
        zero = next(model.parameters()).new_zeros(())
        return zero, zero
    oracle_target = target
    lam = float(config.get('policy_lambda', .01))
    gamma = 0. if variant == 'no_presentation' else float(config.get('policy_gamma', .1))
    values = []
    for example in eligible:
        gain, q, _ = model(torch.as_tensor(example['x']), torch.as_tensor(example['state']),
                           torch.as_tensor(example['candidate']), **candidate_options(example))
        values.append(gain-lam*float(example.get('cost', 0.))-gamma*sensitivity_from_log_variance(q))
    # Binary supervision follows deployment's max(Q)>0 rule and does not split
    # acquisition probability among K candidate identities. Paired gain MSE
    # continues to supervise candidate ranking in this mode.
    if settings['mode'] == 'binary_stop':
        logits = torch.stack([torch.stack(values).max(), values[0].new_zeros(())])
        target = 0 if category == 'acquire' else 1
    else:
        logits = torch.stack(values+[values[0].new_zeros(())])
    logits = logits/settings['temperature']
    cross_entropy = nn.functional.cross_entropy(logits.unsqueeze(0), logits.new_tensor([target], dtype=torch.long))
    ranking = values[0].new_zeros(())
    if settings.get('candidate_ranking_weight', 0.) > 0 and category == 'acquire':
        # Conditional action identity does not compete with STOP. Its target is
        # the same positive, budget-eligible source utility oracle used above.
        candidate_logits = torch.stack(values)/settings['temperature']
        ranking = nn.functional.cross_entropy(candidate_logits.unsqueeze(0),
                    candidate_logits.new_tensor([oracle_target], dtype=torch.long))
    return statistics['class_weights'][category]*cross_entropy, ranking


def decision_choice_loss(model, actions, config, settings, statistics, variant='full'):
    """Compatibility entry point for the existing STOP/acquire loss."""
    return decision_choice_losses(model, actions, config, settings, statistics, variant)[0]


def action_logits(model, actions, config, variant):
    """Deployment Q scores form the trainable action policy; STOP has Q=0."""
    values = []
    lam = float(config.get('policy_lambda', .01))
    gamma = 0. if variant == 'no_presentation' else float(config.get('policy_gamma', .1))
    temperature = float(config['grpo']['temperature'])
    for example in actions:
        gain, q, _ = model(torch.as_tensor(example['x']), torch.as_tensor(example['state']),
                           torch.as_tensor(example['candidate']), **candidate_options(example))
        # Clamp the log-variance before exp; targets also use epsilon=1e-6.
        sensitivity = (q.clamp(-30, 30).exp() - 1e-6).clamp_min(1e-12).sqrt()
        values.append((gain - lam*example.get('cost', 0.) - gamma*sensitivity) / temperature)
    return torch.stack(values + [values[0].new_zeros(())])


def action_rewards(actions, config, variant):
    lam = float(config.get('policy_lambda', .01))
    gamma = 0. if variant == 'no_presentation' else float(config.get('policy_gamma', .1))
    rewards = [e.get('gain', 0.) - lam*e.get('cost', 0.) - gamma*np.sqrt(max(0., e.get('variance', 0.)))
               + e.get('screening_reward_delta', {}).get('total', 0.)
               for e in actions]
    # STOP performs no acquisition: gain, incremental cost and sensitivity are zero.
    return torch.tensor(rewards + [0.], dtype=torch.float32)


def controller_validation(model, examples, config, variant='full'):
    """Participant-balanced gain error and regret of the deployed action-or-STOP choice."""
    lam = float(config.get('policy_lambda', .01))
    gamma = 0. if variant == 'no_presentation' else float(config.get('policy_gamma', .1))
    errors, states, initial_sizes = {}, {}, {}
    for index, example in enumerate(examples):
        if not candidate_action_allowed(example['candidate'], action_filter_mode(config)):
            continue
        state_id = example['state_id']
        record_key = (example['participant'], str(state_id[0]))
        observed_size = len(state_id[1])
        # Determine the sampled initial state before removing over-budget actions.
        # Seed size varies by task; Reader rows in x are not observations.
        initial_sizes[record_key] = min(initial_sizes.get(record_key, observed_size), observed_size)
        gain, q, _ = model(torch.as_tensor(example['x']), torch.as_tensor(example['state']),
                           torch.as_tensor(example['candidate']), **candidate_options(example))
        if not np.isfinite([float(gain), float(q)]).all():
            raise RuntimeError('Non-finite controller validation output')
        participant = example['participant']
        errors.setdefault(participant, []).append(abs(float(gain)-example['gain']))
        cost = float(example.get('cost', 0.))
        # The same normalized cost must fit the remaining deployment budget.
        if cost > float(example['state'][2])*float(config.get('budget', 8)):
            continue
        sensitivity = max(0., math_exp(float(q))-1e-6)**.5
        predicted = float(gain)-lam*cost-gamma*sensitivity
        observed = example['gain']-lam*cost-gamma*np.sqrt(max(0., example['variance']))
        states.setdefault(participant, {}).setdefault(example['state_id'], []).append(
            (predicted, str(example.get('action_id', index)), float(observed)))
    regrets, initial_regrets = {}, {}
    chosen_positive, oracle_positive, state_count = 0, 0, 0
    initial_chosen, initial_oracle, initial_count = 0, 0, 0
    for participant, grouped in states.items():
        for state_id, actions in grouped.items():
            best = max(actions, key=lambda action: (action[0], action[1]))
            oracle = max(0., max(action[2] for action in actions))
            chosen = best[2] if best[0] > 0 else 0.  # STOP has exactly zero utility.
            regrets.setdefault(participant, []).append(oracle-chosen)
            chosen_positive += best[0] > 0
            oracle_positive += oracle > 0
            state_count += 1
            if len(state_id[1]) == initial_sizes[(participant, str(state_id[0]))]:
                initial_regrets.setdefault(participant, []).append(oracle-chosen)
                initial_chosen += best[0] > 0
                initial_oracle += oracle > 0
                initial_count += 1
    return {'gain_mae': float(np.mean([np.mean(v) for v in errors.values()])),
            'utility_regret': float(np.mean([np.mean(v) for v in regrets.values()])) if regrets else 0.,
            'initial_utility_regret': (float(np.mean([np.mean(v) for v in initial_regrets.values()]))
                                       if initial_regrets else None),
            'eligible_states': state_count, 'chosen_acquisitions': chosen_positive,
            'oracle_acquisitions': oracle_positive,
            'initial_eligible_states': initial_count, 'initial_chosen_acquisitions': initial_chosen,
            'initial_oracle_acquisitions': initial_oracle}


def filter_controller_actions(examples, mode):
    original = list(examples or [])
    retained = [e for e in original if candidate_action_allowed(e['candidate'], mode)]
    summary = {'original_pairs': len(original), 'retained_pairs': len(retained),
               'removed_pairs': len(original)-len(retained)}
    if mode != 'none':
        initial, states = {}, {}
        for example in original:
            key = (example['participant'], str(example['state_id'][0]))
            size = len(example['state_id'][1])
            text_count = int((np.asarray(example['x'])[:size, 64] == 1.).sum())
            if key not in initial or size < initial[key][0]:
                initial[key] = (size, text_count)
        for example in retained:
            key = (example['participant'], str(example['state_id'][0]))
            ids = tuple(example['state_id'][1])
            states[(key, ids)] = int((np.asarray(example['x'])[:len(ids), 64] == 1.).sum())
        summary.update(retained_states=len(states), later_states_with_additional_text=sum(
            len(ids) > initial[key][0] and count > initial[key][1]
            for (key, ids), count in states.items()))
    return retained, summary


def participant_order_mode(config):
    mode = config.get('participant_order', 'fixed')
    if mode not in ('fixed', 'seeded_epoch_shuffle'):
        raise ValueError('Unsupported Controller participant order')
    return mode


def participant_group_order(count, mode='fixed', seed=27, epoch=0):
    """Return an epoch's group indices without using any global RNG stream."""
    mode = participant_order_mode({'participant_order': mode})
    if count < 0 or epoch < 0:
        raise ValueError('Controller group count and epoch must be nonnegative')
    indices = list(range(count))
    if mode == 'seeded_epoch_shuffle':
        random.Random(f'ega-participant-order-v1:{int(seed)}:{int(epoch)}').shuffle(indices)
    return indices


def fit(train, dev, config, variant='full', seed=27, selected=None,
        checkpoint_path=None, resume=False):
    if variant not in ('full', 'no_presentation'):
        raise ValueError('Unsupported acquisition-controller variant')
    participant_order = participant_order_mode(config)
    budget_mode = budget_normalization_mode(config)
    budget_profile = fit_budget_profile(train, config)
    budget_statistics = None
    if budget_profile is not None:
        config = dict(config, budget_normalization=budget_mode, budget_profile=budget_profile)
        train = rebase_budget_examples(train, config, budget_profile)
        dev = rebase_budget_examples(dev, config, budget_profile)
        budget_statistics = {name: {'pairs': len(rows),
                                   'original_clamped_pairs': sum(e['budget_rebase']['clamped'] for e in rows),
                                   'rebased_exhausted_pairs': sum(float(e['state'][2]) == 0 for e in rows)}
                             for name, rows in (('train', train), ('dev', dev))}
        print('controller_budget_units: '+json.dumps({'profile': budget_profile,
                                                      'statistics': budget_statistics}), flush=True)
    from .support.controller_states import state_filter_mode, filter_controller_states
    state_filter = state_filter_mode(config)
    train, train_state_filter = filter_controller_states(train, state_filter)
    dev, dev_state_filter = filter_controller_states(dev, state_filter)
    state_filter_statistics = {'mode': state_filter, 'train': train_state_filter, 'dev': dev_state_filter}
    if not train:
        raise ValueError('Controller state filter retained no source training targets')
    if state_filter != 'none' and dev_state_filter['original_pairs'] and not dev:
        raise ValueError('Controller state filter retained no source development targets')
    if state_filter != 'none':
        config = dict(config, state_filter=state_filter, state_filter_statistics=state_filter_statistics)
    action_filter = action_filter_mode(config)
    train, train_filter = filter_controller_actions(train, action_filter)
    dev, dev_filter = filter_controller_actions(dev, action_filter)
    filter_statistics = {'mode': action_filter, 'train': train_filter, 'dev': dev_filter}
    if not train:
        raise ValueError('Controller action filter retained no source training targets')
    if action_filter != 'none' and dev_filter['original_pairs'] and not dev:
        raise ValueError('Controller action filter retained no source development targets')
    sensitivity_loss = config.get('sensitivity_loss', 'log_variance_mse')
    if sensitivity_loss not in ('log_variance_mse', 'std_mse'):
        raise ValueError('Unsupported Controller sensitivity loss')
    choice_settings = decision_config(config)
    from .support.reader_state import reader_state_mode
    evidence_encoding = reader_state_mode(config)
    candidate_mode = candidate_conditioning_mode(config)
    state_evidence_mode = controller_state_evidence_mode(config)
    state_weighting = config.get('training_state_weighting', 'uniform')
    if state_weighting not in ('uniform', 'initial_later_balanced'):
        raise ValueError('Unsupported Controller training state weighting')
    gain_loss = gain_loss_settings(config)
    if 'gain_loss' in config:
        config = dict(config, gain_loss=gain_loss)
    score_profile = _score_profile(config.get('score_calibration'))
    metadata_normalization = config.get('metadata_normalization', 'none')
    if metadata_normalization not in ('none', 'source_clipped_minmax'):
        raise ValueError('Unsupported Controller metadata normalization')
    metadata_bounds = fit_metadata_bounds(train) if metadata_normalization == 'source_clipped_minmax' else None
    if metadata_bounds is not None:
        config = dict(config, metadata_normalization=metadata_normalization, metadata_bounds=metadata_bounds)
    state_representation, seed_anchor = controller_inputs(config)
    def profiled_examples(examples):
        output = []
        for example in examples:
            validate_gain_profile(example, gain_loss, score_profile)
            validate_candidate_mask(example, candidate_mode)
            if reader_state_mode(example) != evidence_encoding:
                raise ValueError('Teacher Controller Reader encoding changed; re-encode source rows')
            if controller_state_evidence_mode(example) != state_evidence_mode:
                raise ValueError('Teacher Controller Reader evidence mode changed; re-encode source states')
            if controller_inputs(example) != (state_representation, seed_anchor):
                raise ValueError('Teacher input/anchor configuration changed; regenerate paired targets')
            if (state_representation == 'frozen_token_embeddings'
                    and example.get('state_representation_version') != STATE_REPRESENTATION_VERSION):
                raise ValueError('Frozen Controller representation version changed; regenerate paired targets')
            if 'cost_features' in example:
                if 'cost_model' not in config:
                    raise ValueError('Fit timing cost_model before controller training; post-action timing is not a policy input')
                cost = cost_from_features(example['cost_features'], config['cost_model'], example.get('cost_basis'))
                example = dict(example, cost=cost/budget_reference(config, example['seed_seconds']))
            output.append(example)
        return output
    train, dev = profiled_examples(train), profiled_examples(dev or [])
    from .support.controller_ranking import ranking_settings, derive_ranking_targets
    ranking_config = ranking_settings(config)
    train, dev, ranking_profile = derive_ranking_targets(train, dev, ranking_config, gain_loss)
    if ranking_config['enabled']:
        config = dict(config, ranking_loss=ranking_config, ranking_profile=ranking_profile)
    grouped = {}
    for example in train:
        grouped.setdefault(example['participant'], {}).setdefault(example['state_id'], []).append(example)
    if not grouped:
        raise ValueError('Controller training requires at least one teacher action')
    choice_statistics = (decision_training_statistics(grouped, config, choice_settings, variant)
                         if choice_settings['enabled'] else None)
    if choice_settings['enabled']:
        # Derived class weights are bound to TRAIN and to this resume signature.
        config = dict(config, decision_loss=choice_settings, decision_training=choice_statistics)
    alpha, beta = float(config.get('alpha', .1)), float(config.get('beta', .1))
    if variant == 'no_presentation':
        alpha, beta = 0., 0.
    if selected is not None:
        alpha, beta = selected['alpha'], selected['beta']
    grpo_config = {'enabled': True, 'weight': 1., 'supervised_weight': 1., 'group_size': 8,
                   'temperature': .1, 'clip_epsilon': .2, 'kl_beta': .04,
                   'updates_per_group': 2, 'max_grad_norm': 1., 'max_steps': None}
    grpo_config.update(config.get('grpo', {}))
    grpo.validate(grpo_config)
    config = dict(config, grpo=grpo_config)
    target_mode = target_normalization_mode(config)
    target_profile = fit_target_profile(grouped, config, alpha, beta, variant)
    if target_profile is not None:
        config = dict(config, target_normalization=target_mode, target_profile=target_profile)
        print('controller_target_units: '+json.dumps(target_profile), flush=True)
    epochs = int(selected['epochs'] if selected else config['epochs'])
    if epochs < 1:
        raise ValueError('Controller epochs must be positive')
    expected = grpo.signature((config, variant, seed, selected, train, dev))
    restored = grpo.load_checkpoint(checkpoint_path, resume, 'ega-controller-training-v1', expected)
    torch.manual_seed(seed)
    model = Controller(metadata_normalization, metadata_bounds, candidate_mode,
                       target_normalization=target_mode, target_profile=target_profile)
    opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    step, history, start_epoch, next_group = 0, [], 0, 0
    best_error, stale, epoch_best, saved, complete = float('inf'), 0, 0, None, False
    best_gain_mae, best_regret, best_initial_regret = float('inf'), float('inf'), float('inf')
    validation_history = []
    if restored:
        if restored.get('participant_order', 'fixed') != participant_order:
            raise ValueError('Restored Controller participant order changed')
        if (restored.get('target_normalization', 'none') != target_mode
                or validate_target_profile(restored.get('target_profile')) != target_profile):
            raise ValueError('Restored Controller target normalization changed')
        model.load_state_dict(restored['controller'])
        opt.load_state_dict(restored['optimizer'])
        reference.load_state_dict(restored['reference'])
        step, history = restored['step'], restored['history']
        start_epoch, next_group = restored['epoch'], restored['next_group']
        best_error, stale, epoch_best = restored['best_error'], restored['stale'], restored['best_epoch']
        best_gain_mae = restored.get('best_gain_mae', best_error)
        best_regret = restored.get('best_regret', float('inf'))
        best_initial_regret = restored.get('best_initial_regret', float('inf'))
        validation_history = restored.get('validation_history', [])
        saved, complete = restored['best_controller'], restored['complete']
        grpo.restore_rng(restored['rng'])
        print(f'controller_resume: epoch={start_epoch+1} next_group={next_group} step={step}', flush=True)

    def checkpoint(epoch, group, done, boundary):
        if checkpoint_path is not None:
            grpo.save_checkpoint(checkpoint_path, {
                'schema': 'ega-controller-training-v1', 'signature': expected,
                'controller': model.state_dict(), 'optimizer': opt.state_dict(),
                'reference': reference.state_dict(), 'rng': grpo.rng_state(),
                'epoch': epoch, 'next_group': group, 'step': step, 'history': history,
                'best_error': best_error, 'stale': stale, 'best_epoch': epoch_best,
                'best_gain_mae': best_gain_mae, 'best_regret': best_regret,
                'best_initial_regret': best_initial_regret,
                'decision_loss': choice_settings, 'decision_training': choice_statistics,
                'training_state_weighting': state_weighting,
                'target_normalization': target_mode, 'target_profile': target_profile,
                'participant_order': participant_order,
                'budget_normalization': budget_mode, 'budget_profile': budget_profile,
                'budget_rebase_statistics': budget_statistics,
                'state_evidence': state_evidence_mode,
                'candidate_conditioning': candidate_mode,
                'action_filter': action_filter, 'action_filter_statistics': filter_statistics,
                'state_filter': state_filter, 'state_filter_statistics': state_filter_statistics,
                'ranking_loss': ranking_config, 'ranking_profile': ranking_profile,
                'state_evidence_encoding': evidence_encoding,
                'gain_loss': gain_loss,
                'metadata_normalization': metadata_normalization, 'metadata_bounds': metadata_bounds,
                'validation_history': validation_history,
                'best_controller': saved, 'complete': done, 'boundary': boundary})

    validation = [e for e in (dev or train) if e.get('supervised_valid', True)]
    requested_selection = config.get('selection_metric', 'gain_mae')
    if requested_selection not in ('gain_mae', 'utility_regret', 'initial_utility_regret'):
        raise ValueError('Controller selection_metric must be gain_mae, utility_regret, or initial_utility_regret')
    if requested_selection in ('utility_regret', 'initial_utility_regret') and not validation:
        raise ValueError('STOP-aware regret selection requires valid paired development targets')
    selection = requested_selection if validation else 'negative_expected_reward'
    validation_groups = {}
    if not validation:
        for example in (dev or train):
            validation_groups.setdefault(example['participant'], {}).setdefault(example['state_id'], []).append(example)
    groups = list(grouped.values())
    for epoch in range(start_epoch, epochs) if not complete else ():
        model.train()
        group_order = participant_group_order(len(groups), participant_order, seed, epoch)
        for group_index in range(next_group if epoch == start_epoch else 0, len(groups)):
            if grpo_config['max_steps'] is not None and step >= int(grpo_config['max_steps']):
                break
            # next_group is a position in this reproducible epoch permutation.
            states = groups[group_order[group_index]]
            state_weights = training_state_weights(states, state_weighting)
            sampled_groups = []
            if grpo_config['enabled']:
                with torch.no_grad():
                    for actions in states.values():
                        sampled_groups.append(grpo.rollout(
                            action_logits(model, actions, config, variant), action_rewards(actions, config, variant),
                            action_logits(reference, actions, config, variant), grpo_config['group_size']))
            # One behavior snapshot is retained for all updates in this group.
            iterations = grpo_config['updates_per_group'] if grpo_config['enabled'] else 1
            if grpo_config['max_steps'] is not None:
                iterations = min(iterations, int(grpo_config['max_steps'])-step)
            for _ in range(iterations):
                opt.zero_grad()
                state_losses, stats, choice_losses, ranking_losses = [], [], [], []
                for index, actions in enumerate(states.values()):
                    valid_actions = [e for e in actions if e.get('supervised_valid', True)]
                    terms = [loss(model, e, alpha, beta, sensitivity_loss) for e in valid_actions]
                    supervised = (torch.stack([m for m, _, _ in terms]).mean() + terms[0][1]
                                  if terms else next(model.parameters()).new_zeros(()))
                    state_loss = grpo_config['supervised_weight'] * supervised
                    if choice_settings['enabled']:
                        choice, ranking = decision_choice_losses(
                            model, actions, config, choice_settings, choice_statistics, variant)
                        weighted_choice = choice_settings['weight']*choice
                        weighted_ranking = choice_settings['candidate_ranking_weight']*ranking
                        state_loss = state_loss + weighted_choice + weighted_ranking
                        choice_losses.append(float(weighted_choice.detach()))
                        ranking_losses.append(float(weighted_ranking.detach()))
                    if grpo_config['enabled']:
                        policy_loss, statistic = grpo.objective(
                            action_logits(model, actions, config, variant), sampled_groups[index],
                            grpo_config['clip_epsilon'], grpo_config['kl_beta'])
                        state_loss = state_loss + grpo_config['weight'] * policy_loss
                        stats.append(statistic)
                    state_losses.append(state_loss)
                if state_weighting == 'initial_later_balanced':
                    objective_weights = [state_weights[state_id] for state_id in states]
                    total = sum(value*weight for value, weight in zip(state_losses, objective_weights))
                else:
                    total = torch.stack(state_losses).mean()
                if not torch.isfinite(total) or not total.requires_grad:
                    raise RuntimeError('Controller has no finite trainable objective')
                total.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grpo_config['max_grad_norm'],
                                                              error_if_nonfinite=True)
                opt.step()
                step += 1
                summary = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]} if stats else {}
                summary.update(step=step, loss=float(total.detach()), gradient_norm=float(gradient_norm))
                if choice_settings['enabled']:
                    summary['decision_loss'] = (float(np.dot(choice_losses, objective_weights))
                                                if state_weighting == 'initial_later_balanced'
                                                else float(np.mean(choice_losses)))
                    if choice_settings['candidate_ranking_weight']:
                        summary['candidate_ranking_loss'] = (float(np.dot(ranking_losses, objective_weights))
                            if state_weighting == 'initial_later_balanced' else float(np.mean(ranking_losses)))
                history.append(summary)
                print('controller_update: '+str(summary), flush=True)
            # No in-flight rollout is needed when resuming from this boundary.
            checkpoint(epoch, group_index+1, False, 'completed_policy_group')
        model.eval()
        with torch.no_grad():
            errors = {}
            if validation:
                validation_stat = controller_validation(model, validation, config, variant)
                error = validation_stat[selection]
                if error is None:
                    raise ValueError('Initial regret selection requires a budget-eligible sampled initial state')
                gain_mae = validation_stat['gain_mae']
                regret = validation_stat['utility_regret']
                initial_regret = validation_stat['initial_utility_regret']
            else:
                for participant, states in validation_groups.items():
                    errors[participant] = [float(-(action_logits(model, actions, config, variant).softmax(0)
                                                  * action_rewards(actions, config, variant)).sum())
                                           for actions in states.values()]
                error = float(np.mean([np.mean(v) for v in errors.values()]))
                gain_mae, regret, initial_regret = float('inf'), float('inf'), None
                validation_stat = {'negative_expected_reward': error}
        validation_history.append({'epoch': epoch+1, 'selection_metric': selection, **validation_stat})
        print('controller_validation: '+str(validation_history[-1]), flush=True)
        if selected is not None or (error, gain_mae) < (best_error, best_gain_mae):
            best_error, stale, saved, epoch_best = error, 0, copy.deepcopy(model.state_dict()), epoch+1
            best_gain_mae, best_regret = gain_mae, regret
            best_initial_regret = initial_regret if initial_regret is not None else float('inf')
        else:
            stale += 1
        complete = ((selected is None and stale >= config['patience']) or epoch+1 >= epochs
                    or (grpo_config['max_steps'] is not None and step >= int(grpo_config['max_steps'])))
        checkpoint(epoch+1, 0, complete, 'completed_epoch')
        if complete:
            break
    if saved is None:
        raise ValueError('Controller checkpoint contains no completed validation epoch')
    gain_mae = best_gain_mae if np.isfinite(best_gain_mae) else None
    trials = [{'alpha': alpha, 'beta': beta, 'dev_gain_mae': gain_mae, 'epochs': epoch_best,
               'optimizer_steps': step}]
    result = Controller.from_state_dict(saved, target_profile=target_profile)
    result.eval().requires_grad_(False)
    parameter_delta = float(torch.sqrt(sum((p.detach()-r.detach()).square().sum()
                                          for p, r in zip(result.parameters(), reference.parameters()))))
    return result, {'alpha': alpha, 'beta': beta, 'dev_gain_mae': gain_mae, 'seed': seed,
                    'variant': variant, 'epochs': epoch_best, 'trials': trials,
                    'selection_metric': selection, 'dev_selection_error': best_error,
                    'dev_utility_regret': best_regret if np.isfinite(best_regret) else None,
                    'dev_initial_utility_regret': (best_initial_regret if np.isfinite(best_initial_regret) else None),
                    'validation_history': validation_history,
                    'resumed': restored is not None,
                    'sensitivity_loss': sensitivity_loss,
                    'gain_loss': gain_loss,
                    **({'score_calibration': score_profile} if score_profile is not None else {}),
                    'metadata_normalization': metadata_normalization, 'metadata_bounds': metadata_bounds,
                    'decision_loss': choice_settings, 'decision_training': choice_statistics,
                    'training_state_weighting': state_weighting,
                    'target_normalization': target_mode, 'target_profile': target_profile,
                    'participant_order': participant_order,
                    'budget_normalization': budget_mode, 'budget_profile': budget_profile,
                    'budget_rebase_statistics': budget_statistics,
                    'state_evidence': state_evidence_mode,
                    'candidate_conditioning': candidate_mode,
                    'action_filter': action_filter, 'action_filter_statistics': filter_statistics,
                    'state_filter': state_filter, 'state_filter_statistics': state_filter_statistics,
                    'ranking_loss': ranking_config, 'ranking_profile': ranking_profile,
                    'state_evidence_encoding': evidence_encoding,
                    'algorithm': 'GRPO+paired_supervision' if grpo_config['enabled'] else 'paired_supervision',
                    'optimizer': 'grpo' if grpo_config['enabled'] else 'adam',
                    'grpo': grpo_config, 'optimizer_steps': step, 'updates': history,
                    'grpo_history': history if grpo_config['enabled'] else [],
                    'policy_lambda': float(config.get('policy_lambda', .01)),
                    'policy_gamma': 0. if variant == 'no_presentation' else float(config.get('policy_gamma', .1)),
                    'parameter_delta_norm': parameter_delta}


def fit_cost(timings):
    from scipy.optimize import nnls
    if not timings:
        raise ValueError('Acquisition cost fitting requires measured timings')
    new_basis = all(len(row) == 4 and row[3] == COST_BASIS for row in timings)
    if not new_basis and any(len(row) != 3 for row in timings):
        raise ValueError('Cannot mix acquisition timing bases')
    array = np.asarray([row[:3] for row in timings], dtype=float)
    coefficients, _ = nnls(np.column_stack([np.ones(len(array)), array[:, :2]]), array[:, 2])
    return ({'basis': COST_BASIS, 'coefficients': coefficients.tolist()} if new_basis else coefficients.tolist())


def predicted_cost(views, coefficients):
    basis = coefficients['basis'] if isinstance(coefficients, dict) else 'legacy_total_duration'
    return cost_from_features(cost_features(views, basis), coefficients, basis)


def acquire(backbone, rec, track, model, policy, budget, limit, variant='full'):
    started = time.perf_counter()-rec.get('index_seconds', 0.)
    state = seed_views(rec, track)
    directory = pool(rec, track)
    plain_score = getattr(backbone, 'score_unaugmented', backbone.score)
    state_representation, seed_anchor = controller_inputs(policy)
    candidate_mode = candidate_conditioning_mode(policy)
    action_filter = action_filter_mode(policy)
    from .support.reader_state import reader_state_mode
    evidence_encoding = reader_state_mode(policy)
    from .support.controller_states import state_filter_mode
    if state_filter_mode(policy) == 'seed_aligned_audio' and action_filter != 'aligned_audio':
        raise ValueError('Seed-aligned Controller states require aligned-audio acquisition')
    if candidate_mode != model.candidate_conditioning:
        raise ValueError('Controller checkpoint and runtime candidate-conditioning policy differ')
    capture_context = state_representation == 'frozen_discriminator_context'
    capture_options = {'capture_controller_state': True} if capture_context else {}
    score_profile = _score_profile(policy.get('score_calibration'))
    runtime_options = dict(capture_options, **({'score_calibration': score_profile} if score_profile is not None else {}))
    initial = calibrate_prediction(plain_score(state, **capture_options), state, score_profile)
    seed_cost = max(1e-6, initial['seconds']+rec.get('index_seconds', 0.))
    reference_seconds = budget_reference(policy, seed_cost)
    if budget_normalization_mode(policy) != 'per_seed_wall' and float(policy['budget_profile']['budget']) != float(budget):
        raise ValueError('Runtime budget differs from fitted source budget multiplier')
    total_limit = reference_seconds * budget
    reader_gate = policy.get('reader_gate')
    use_reader = reader_allowed(backbone, reader_gate)
    state_evidence_mode = controller_state_evidence_mode(policy)
    use_state_reader = use_reader or state_evidence_mode == 'observed_reader'
    plain_anchor = seed_anchor == 'plain_until_acquisition'
    if plain_anchor:
        if time.perf_counter()-started < total_limit and initial['status'] == 'ok':
            state_evidence, state_meta = controller_state_evidence(
                backbone, state, reader_gate, state_evidence=state_evidence_mode)
        else:
            state_evidence, state_meta = [], {'state_reader_used': False,
                                            'state_reader_decision': 'initial_budget_or_score_failure',
                                            'state_reader_seconds': 0.,
                                            'reader_gate_signature': reader_gate_signature(reader_gate)}
        current = initial | {'reader_used': False, 'reader_decision': 'plain_until_acquisition', **state_meta}
        initial_reader_decision = state_meta['state_reader_decision']
    elif use_reader:
        if time.perf_counter()-started < total_limit and initial['status'] == 'ok':
            current = score_with_reader_gate(backbone, state, reader_gate, plain=initial,
                                             **runtime_options)
        else:
            current = initial | {'reader_used': False, 'reader_decision': 'initial_budget_or_score_failure',
                                 'reader_gate_signature': reader_gate_signature(reader_gate)}
    else:
        current = score_with_reader_gate(backbone, state, reader_gate, plain=initial, **runtime_options)
    if not plain_anchor:
        initial_reader_decision = current['reader_decision']
        state_evidence, state_meta = controller_state_evidence(
            backbone, state, reader_gate, prediction=current, state_evidence=state_evidence_mode)
    if state_evidence_mode == 'observed_reader':
        current = current | state_meta | {'state_reader_records': state_evidence,
                                         'seconds': current['seconds']+state_meta['state_reader_seconds']}
    trace, decisions, reader_diagnostic_work = [], [], []
    state_history_work = [(0, tuple(view['id'] for view in state), state_evidence,
                           current['p'], current.get('raw_p', current['p']), current['reader_used'])]
    stop_reason = 'acquisition_limit'
    for _ in range(limit):
        if current['status'] != 'ok':
            stop_reason = 'score_failure'
            break
        elapsed = time.perf_counter()-started
        remaining = total_limit-elapsed
        if remaining <= 0:
            stop_reason = 'budget_exhausted'
            break
        x = torch.tensor(encode_state(state, state_evidence, backbone, state_representation,
                                      controller_context=current.get('controller_context'),
                                      state_evidence_encoding=evidence_encoding))
        state_input = torch.tensor([current['p'], 1., remaining/total_limit], dtype=torch.float32)
        choices = []
        summary = {'step': len(trace), 'remaining_seconds': remaining, 'considered': 0,
                   'rejected': {'reader_positions': 0, 'context': 0, 'cost': 0, 'nonfinite_controller': 0}}
        if action_filter != 'none':
            summary['rejected']['action_filter'] = 0
        with torch.no_grad():
            for action in directory:
                if action in state:
                    continue
                summary['considered'] += 1
                candidate = metadata(action, state)
                if not candidate_action_allowed(candidate, action_filter):
                    summary['rejected']['action_filter'] += 1
                    continue
                reader = getattr(backbone, 'reader', None)
                if use_state_reader:
                    if reader.candidate_capacity_exceeded(state, action):
                        summary['rejected']['reader_positions'] += 1
                        continue
                if hasattr(backbone, 'protocol'):
                    reserved = action.get('context_reservation')
                    if reserved is None:
                        raise ValueError('Candidate lacks an indexed context reservation')
                    if use_reader:
                        # Rereading can change the records of retained observations as well.
                        reserved += max(0, backbone.protocol.get('max_evidence_tokens', 1024)-current.get('evidence_tokens', 0))
                    if current['input_tokens']+reserved > backbone.protocol['max_input_tokens']:
                        summary['rejected']['context'] += 1
                        continue
                cost = predicted_cost(state+[action], policy['cost'])
                if cost > total_limit-(time.perf_counter()-started):
                    summary['rejected']['cost'] += 1
                    continue
                mask = encode_candidate_mask(state, state_evidence, action, state_representation, candidate_mode)
                gain, q, _ = model(x, state_input, torch.tensor(candidate),
                                   **({'candidate_mask': mask} if mask is not None else {}))
                if not np.isfinite([float(gain), float(q)]).all():
                    summary['rejected']['nonfinite_controller'] += 1
                    continue
                sensitivity = max(0., math_exp(float(q))-1e-6)**.5
                value = float(gain)-policy['lambda']*cost/reference_seconds-policy['gamma']*sensitivity
                details = {'view': action['id'], 'gain': float(gain), 'sensitivity': sensitivity,
                           'cost_penalty': policy['lambda']*cost/reference_seconds,
                           'sensitivity_penalty': policy['gamma']*sensitivity,
                           'value': value, 'estimated_seconds': cost}
                if mask is not None:
                    details.update(candidate_conditioning=candidate_mode, matched_observed_rows=int(mask.sum()))
                choices.append((value, action['id'], action, cost, details))
        summary.update(eligible=len(choices), positive_utility=sum(c[0] > 0 for c in choices))
        if policy.get('reader_diagnostics', False) and state_evidence and choices:
            # Save already-observed inputs; run this local comparison only after
            # acquisition ends, so diagnostics cannot consume the policy budget.
            reader_diagnostic_work.append((summary, x, state_input, len(state), len(state_evidence),
                [(c[2], metadata(c[2], state), encode_candidate_mask(
                    state, state_evidence, c[2], state_representation, candidate_mode), c[4]) for c in choices]))
        decisions.append(summary)
        if not choices:
            stop_reason = 'no_candidates' if not summary['considered'] else 'no_eligible_candidates'
            break
        value, _, action, cost, details = max(choices, key=lambda x: (x[0], x[1]))
        summary['best'] = details
        if value <= 0:
            stop_reason = 'nonpositive_utility'
            break
        if cost > total_limit-(time.perf_counter()-started):
            stop_reason = 'selected_cost_exceeds_remaining'
            break
        state = canonical(state+[action])
        current = score_with_reader_gate(backbone, state, reader_gate, **runtime_options)
        state_evidence, state_meta = controller_state_evidence(
            backbone, state, reader_gate, prediction=current, state_evidence=state_evidence_mode)
        if state_evidence_mode == 'observed_reader':
            current = current | state_meta | {'state_reader_records': state_evidence,
                                             'seconds': current['seconds']+state_meta['state_reader_seconds']}
        trace.append(details | {'p': current['p'], 'seconds': current['seconds'], 'status': current['status'],
                                'reader_used': current['reader_used'], 'reader_decision': current['reader_decision'],
                                'state_reader_used': bool(state_evidence)})
        state_history_work.append((len(trace), tuple(view['id'] for view in state), state_evidence,
                                   current['p'], current.get('raw_p', current['p']), current['reader_used']))
        if current['status'] != 'ok':
            stop_reason = 'score_failure'
            break
    elapsed = time.perf_counter()-started
    observed_lookup = {view['id']: view for view in rec['views']}
    state_history = [{'step': step, 'observed_view_ids': list(view_ids),
                      'reader_records': copy.deepcopy(records), 'p': probability, 'raw_p': raw_probability,
                      'classifier_reader_used': classifier_reader_used,
                      'observed_text': {identity: observed_lookup[identity]['text'] for identity in view_ids
                                        if 'text' in observed_lookup[identity]},
                      'text_basis': 'Original already-observed text; not a Reader-extracted span.'}
                     for step, view_ids, records, probability, raw_probability, classifier_reader_used
                     in state_history_work]
    reader_diagnostic_seconds = 0.
    if reader_diagnostic_work:
        from .support.reader_state import strip_reader_rows
        diagnostic_started = time.perf_counter()
        with torch.no_grad():
            for summary, observed_x, input_state, observed_count, evidence_count, candidates in reader_diagnostic_work:
                without = []
                for action, candidate, candidate_mask, details in candidates:
                    reduced_x, reduced_mask = strip_reader_rows(
                        observed_x.numpy(), observed_count, evidence_count, candidate_mask)
                    gain, q, _ = model(torch.as_tensor(reduced_x), input_state, torch.as_tensor(candidate),
                        **({'candidate_mask': reduced_mask} if reduced_mask is not None else {}))
                    sensitivity = max(0., math_exp(float(q))-1e-6)**.5
                    value = float(gain)-details['cost_penalty']-policy['gamma']*sensitivity
                    if not np.isfinite([float(gain), value]).all():
                        raise RuntimeError('Non-finite Reader feature diagnostic')
                    details.update(gain_without_reader=float(gain), value_without_reader=value,
                                   reader_value_delta=details['value']-value)
                    without.append((value, action['id']))
                actual = max(candidates, key=lambda item: (item[3]['value'], item[0]['id']))
                removed = max(without, key=lambda item: (item[0], item[1]))
                actual_action = actual[0]['id'] if actual[3]['value'] > 0 else 'STOP'
                removed_action = removed[1] if removed[0] > 0 else 'STOP'
                summary['reader_feature_diagnostic'] = {
                    'records': evidence_count, 'actual_action': actual_action,
                    'without_reader_action': removed_action, 'decision_changed': actual_action != removed_action,
                    'actual_best_value': actual[3]['value'], 'without_reader_best_value': removed[0],
                    'scope': 'Local feature removal on the same observed state; no new Reader/LLM calls or trajectory.'}
                summary['candidate_values'] = [copy.deepcopy(item[3]) for item in candidates]
        reader_diagnostic_seconds = time.perf_counter()-diagnostic_started
    return initial, current | {'seconds': elapsed, 'gpu_seconds': elapsed if backbone.device.startswith('cuda') else 0.,
                              **({'budget_normalization': 'source_train_seed_median',
                                  'budget_profile': copy.deepcopy(policy['budget_profile']),
                                  'budget_reference_seconds': reference_seconds, 'seed_seconds': seed_cost}
                                 if budget_normalization_mode(policy) != 'per_seed_wall' else {}),
                              'reader_diagnostic_seconds': reader_diagnostic_seconds,
                              'trace': trace, 'steps': len(trace), 'budget_seconds': total_limit,
                              'overrun_seconds': max(0., elapsed-total_limit), 'stop_reason': stop_reason,
                              'candidate_summary': decisions, 'initial_reader_decision': initial_reader_decision,
                              'state_history': state_history,
                              'state_representation': state_representation,
                              'state_representation_version': STATE_REPRESENTATION_VERSION,
                              'seed_prediction_anchor': seed_anchor,
                              'state_evidence': state_evidence_mode,
                              'state_evidence_encoding': evidence_encoding,
                              'candidate_conditioning': candidate_mode,
                              'action_filter': action_filter}


def math_exp(value):
    return float(np.exp(np.clip(value, -30, 30)))


def select_policy(backbone, records, track, controller, cost, budget, limit, variant='full', policy_context=None):
    choices = []
    gammas = [0.] if variant == 'no_presentation' else [0., .1, .5]
    for lam, gamma in itertools.product((0., .01, .1, 1.), gammas):
        policy = {**(policy_context or {}), 'lambda': lam, 'gamma': gamma, 'cost': cost}
        losses, steps, seconds = [], [], []
        for rec in records:
            _, post = acquire(backbone, rec, track, controller, policy, budget, limit, variant)
            losses.append((post['p']-rec['label'])**2)
            steps.append(post['steps'])
            seconds.append(post['seconds'])
        choices.append((float(np.mean(losses)), lam, gamma, float(np.mean(steps)), float(np.mean(seconds)), policy))
    best = min(choices, key=lambda x: (x[0], x[1], x[2]))
    return best[-1] | {'dev_brier': best[0], 'dev_mean_steps': best[3], 'dev_mean_seconds': best[4],
                      'trials': [{'lambda': c[1], 'gamma': c[2], 'dev_brier': c[0]} for c in choices]}
