"""Frozen source-position representations and validated Reader supervision."""
import hashlib
import math

import numpy as np
import torch

from .data import canonical, read_jsonl
from .models import InputError, waveform


FIELDS = ('speaker', 'experiencer', 'time', 'polarity', 'degree', 'status')


def hashed_token(text):
    vector = np.zeros(64, dtype=np.float32)
    # Fixed character features also support unseen words; no task labels are used.
    for token in [text.casefold()] + list(text.casefold()):
        value = hashlib.blake2b(token.encode(), digest_size=4).digest()
        vector[int.from_bytes(value[:2], 'little') % 64] += 1 if value[2] % 2 else -1
    return vector / max(1., float(np.linalg.norm(vector)))


class SourceEncoder:
    """No fitted parameters. Text uses frozen backbone token embeddings.

    Audio uses 100 ms waveform frames; features retain exposed temporal rows.
    These lightweight media representations do not claim pretrained semantics.
    """
    def __init__(self, backbone, config):
        self.backbone, self.config = backbone, config

    def encode(self, view):
        kind = view['modality']
        coordinates, excerpts, values, offsets = [], [], [], None
        if kind == 'T':
            tokenizer = self.backbone.tokenizer
            ids = tokenizer.encode(view['text'], add_special_tokens=False)
            if not ids:
                raise InputError('Reader received empty text tokens')
            if len(ids) > self.config['max_positions']:
                raise InputError('Reader text exceeds position limit; truncation is forbidden')
            embedding = self.backbone.model.get_input_embeddings()
            with torch.no_grad():
                raw = embedding(torch.tensor(ids, device=embedding.weight.device)).float()
                values = torch.nn.functional.adaptive_avg_pool1d(raw.unsqueeze(1), 64).squeeze(1).cpu().numpy()
            coordinates = [(i, i+1) for i in range(len(ids))]
            excerpts = [tokenizer.decode([i]) for i in ids]
            if getattr(tokenizer, 'is_fast', False):
                encoded = tokenizer(view['text'], add_special_tokens=False, return_offsets_mapping=True)
                if list(encoded['input_ids']) != ids:
                    raise InputError('Reader tokenizer offsets disagree with source token ids')
                offsets = encoded['offset_mapping']
        elif kind == 'A':
            wave = waveform(view)
            size = int(round(self.config['audio_frame_seconds']*16000))
            for first in range(0, len(wave), size):
                frame = wave[first:first+size]
                pieces = np.array_split(frame, min(32, len(frame)))
                summary = np.zeros(64, dtype=np.float32)
                for j, piece in enumerate(pieces):
                    summary[2*j:2*j+2] = piece.mean(), np.sqrt(np.mean(piece**2))
                values.append(summary)
                coordinates.append((view['start']+first/16000,
                                    min(view['stop'], view['start']+(first+len(frame))/16000)))
        elif kind == 'F':
            # Each exposed modality/frame is a distinct coordinate, never average over time.
            for name, source in sorted(view['features'].items()):
                array = np.load(source['path'], mmap_mode='r', allow_pickle=False)
                clip = np.asarray(array[source['first']:source['last']], dtype=np.float32)
                if clip.ndim != 2 or not len(clip) or not np.isfinite(clip).all():
                    raise InputError('Invalid temporal features for evidence reader')
                for j, frame in enumerate(clip):
                    values.append(np.array([a.mean() if len(a) else 0. for a in np.array_split(frame, 64)], dtype=np.float32))
                    coordinates.append((name, source['first']+j, source['first']+j+1))
        if not len(values) or len(values) > self.config['max_positions']:
            raise InputError('Reader source has zero or too many positions; truncation is forbidden')
        values = np.asarray(values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise InputError('Non-finite reader representation')
        scale = np.maximum(1., np.linalg.norm(values, axis=1, keepdims=True))
        return {'view': view, 'values': values/scale, 'coordinates': coordinates, 'excerpts': excerpts,
                'offsets': offsets}


SPEAKER_NORMALIZATION_MODES = ('literal', 'ontology_casefold_v1')


def normalize_relation_temperatures(temperatures=None):
    """Optional decoder-only source calibration; only degree is supported."""
    if temperatures is None:
        return {}
    if not isinstance(temperatures, dict) or set(temperatures)-{'degree'}:
        raise ValueError('Reader relation_temperatures only permits degree')
    result = {}
    for field, value in temperatures.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError('Reader degree temperature must be finite and positive')
        result[field] = float(value)
    return result


def reader_training_settings(config):
    """Resolve optimizer grouping without inserting defaults into legacy configs."""
    mode = config.get('training_mode', 'fullbatch')
    if mode not in ('fullbatch', 'participant_minibatch_v1'):
        raise ValueError('Unsupported Reader training_mode')
    size = config.get('participant_batch_size')
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size <= 0):
        raise ValueError('Reader participant_batch_size must be a positive integer, not bool')
    if mode == 'participant_minibatch_v1' and size is None:
        raise ValueError('Reader participant_minibatch_v1 requires participant_batch_size')
    return {'mode': mode, 'participant_batch_size': size if mode != 'fullbatch' else None}


def normalize_speaker(speaker, ontology, speaker_normalization='literal'):
    """Shared decoder/feature metadata contract; never infer missing identity."""
    if speaker_normalization not in SPEAKER_NORMALIZATION_MODES:
        raise ValueError('Unsupported Reader speaker_normalization')
    if speaker_normalization == 'literal':
        return speaker if speaker in ontology['speaker'] else 'unknown'
    if not isinstance(speaker, str):
        return 'unknown'
    canonical_names = {name.strip().casefold(): name for name in ontology['speaker']}
    if len(canonical_names) != len(ontology['speaker']):
        raise ValueError('Ambiguous casefolded Reader speaker ontology')
    return canonical_names.get(speaker.strip().casefold(), 'unknown')


def validate_config(config):
    normalize_relation_temperatures(config.get('relation_temperatures'))
    reader_training_settings(config)
    ontology = config['ontology']
    if set(ontology) != set(FIELDS):
        raise ValueError('Reader ontology must define all six contextual fields')
    for key, values in ontology.items():
        if not isinstance(values, list) or 'unknown' not in values or len(values) != len(set(values)):
            raise ValueError(f'Invalid reader ontology: {key}')
    for key in ('epochs', 'patience', 'learning_rate', 'max_positions', 'audio_frame_seconds'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'Reader {key} must be positive')
    if not 0 <= config['consistency_weight'] or not config['presence_thresholds']:
        raise ValueError('Invalid reader consistency weight or thresholds')
    if any(not 0 < t < 1 for t in config['presence_thresholds']):
        raise ValueError('Reader presence thresholds must lie in (0, 1)')
    if config.get('boundary_likelihood', 'legacy_independent') not in (
            'legacy_independent', 'ordered_conditional_stop_v1', 'global_legal_joint_v1'):
        raise ValueError('Unsupported Reader boundary_likelihood')
    if config.get('speaker_normalization', 'literal') not in SPEAKER_NORMALIZATION_MODES:
        raise ValueError('Unsupported Reader speaker_normalization')
    if config.get('memory_modalities', 'all_observed') not in ('all_observed', 'supervised'):
        raise ValueError('Unsupported Reader memory_modalities')


def annotations_for(records, annotations, config):
    """Join by source family AND record id; select only the caller's participants."""
    lookup = {(r['family'], r['id']): r for r in records}
    selected, seen = [], set()
    for item in annotations:
        key = item['family'], item['record_id']
        if key not in lookup:
            continue
        rec = lookup[key]
        if item['participant'] != rec['participant'] or item['split'] != rec.get('split', 'pool'):
            raise ValueError('Reader annotation participant/partition mismatch')
        item_key = key + (item['view_id'],)
        if item_key in seen:
            raise ValueError('Duplicate reader window annotation')
        seen.add(item_key)
        item = dict(item)
        item.setdefault('annotation_complete', item.get('fully_reviewed', False))
        if not isinstance(item['annotation_complete'], bool):
            raise ValueError('annotation_complete must be a boolean')
        views = {v['id']: v for v in rec['views']}
        ids = item.get('context_view_ids', [item['view_id']])
        if item['view_id'] not in ids or len(ids) != len(set(ids)) or any(i not in views for i in ids):
            raise ValueError('Invalid reader source/context references')
        if any(not views[i].get('available', True) for i in ids):
            raise ValueError('Reader annotations refer to an unavailable observation')
        targets = item['records']
        if len(targets) > 8:
            raise ValueError('Reader window exceeds eight records; use smaller aligned windows')
        for target in targets:
            if not isinstance(target['start'], int) or not isinstance(target['stop'], int) or not 0 <= target['start'] < target['stop']:
                raise ValueError('Reader boundaries must be ordered, zero-based position indices')
            if target.get('unit') not in ('character', 'position'):
                raise ValueError('Reader annotations must explicitly declare character or position boundaries')
            if set(target['fields']) - set(FIELDS):
                raise ValueError('Unknown reader relation field')
            for field, value in target['fields'].items():
                if value not in config['ontology'][field]:
                    raise ValueError(f'Invalid reader relation category: {field}={value}')
        if not targets and not item['annotation_complete']:
            continue  # An incomplete window supplies no negative supervision.
        selected.append({'participant': (rec['family'], rec['participant']),
                         'views': canonical([views[i] for i in ids]), 'annotation': item})
    return selected


def load_annotations(path):
    return read_jsonl(path)
