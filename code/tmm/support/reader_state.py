"""Typed Controller features from saved Reader records; no media or model reads."""
import json

import numpy as np

from .reader_inputs import FIELDS, hashed_token, normalize_speaker


FIELD_ORDER = tuple(sorted(FIELDS))
READER_STATE_VERSION = 1
READER_SOFT_STATE_VERSION = 2
RELATION_PROBABILITY_BASIS = 'reader_relation_softmax_v1'
SPAN_BASIS = 'indexed_reader_position_capacity_v1'


def reader_state_mode(config):
    mode = config.get('state_evidence_encoding', 'hash')
    if mode not in ('hash', 'structured', 'structured_soft'):
        raise ValueError('Unsupported Controller Reader evidence encoding')
    return mode


def reader_state_profile(config):
    """Runtime payload policy is independent of the frozen Reader weight identity."""
    mode = reader_state_mode(config)
    profile = {'mode': mode, 'version': READER_STATE_VERSION}
    if mode == 'structured_soft':
        profile.update(version=READER_SOFT_STATE_VERSION, relation_probability_basis=RELATION_PROBABILITY_BASIS)
    return profile


def _confidence(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError('Reader confidence must be numeric')
    value = float(value)
    if not np.isfinite(value) or not 0. <= value <= 1.:
        raise ValueError('Reader confidence must lie in [0, 1]')
    return value


def _soft_relations(record, view, ontology, speaker_normalization='literal'):
    """Validate every category, without reconstructing probabilities from maxima."""
    payload = record.get('relation_probabilities')
    supported = record.get('relation_probability_supported')
    if (record.get('relation_probability_basis') != RELATION_PROBABILITY_BASIS
            or not isinstance(payload, dict) or set(payload) != set(FIELD_ORDER)
            or not isinstance(supported, (list, tuple))
            or any(not isinstance(name, str) for name in supported)
            or len(supported) != len(set(supported)) or not set(supported).issubset(FIELD_ORDER)):
        raise ValueError('Soft Reader state requires the complete declared relation probability payload')
    vectors = {}
    for name in FIELD_ORDER:
        categories = sorted(ontology[name])
        probabilities = payload[name]
        if not isinstance(probabilities, dict) or set(probabilities) != set(categories):
            raise ValueError('Soft Reader probabilities must cover the exact frozen ontology')
        vector = np.asarray([_confidence(probabilities[category]) for category in categories], dtype=np.float32)
        if not np.isclose(float(vector.sum()), 1., rtol=0., atol=1e-5):
            raise ValueError('Soft Reader category probabilities must sum to one')
        # Identity is observed metadata; unsupported relations convey no learned semantics.
        fixed = None
        if name == 'speaker':
            fixed = normalize_speaker(view.get('speaker', 'unknown'), ontology, speaker_normalization)
        elif name not in supported:
            fixed = 'unknown'
        if fixed is not None:
            expected = np.asarray([float(category == fixed) for category in categories], dtype=np.float32)
            if not np.allclose(vector, expected, rtol=0., atol=1e-6):
                raise ValueError('Soft Reader metadata/unsupported fields must retain their exact one-hot meaning')
        vectors[name] = vector
    return vectors


def _span(record, capacity):
    first, stop = record['start'], record['stop']
    if (isinstance(first, bool) or isinstance(stop, bool)
            or not isinstance(first, (int, np.integer)) or not isinstance(stop, (int, np.integer))
            or not 0 <= first < stop <= capacity):
        raise ValueError('Reader span exceeds its indexed observed source positions')
    return [float(first)/capacity, float(stop)/capacity]


def encode_reader_record(record, view, ontology, mode='hash', speaker_normalization='literal'):
    """Return 64 features; structured rows have an explicit evidence marker at 63.

    Categorical coordinates use sorted field names and category names, followed
    by six relation confidences, presence/span/joint confidence, context and
    predicted-span positions, and a diagnostic-availability flag. Positions are
    relative to the indexed source capacity, not a guessed tokenizer length.
    structured_soft uses the same categorical slots for full probabilities;
    fields and confidence thresholds remain the Reader decoder's original output.
    """
    mode = reader_state_mode({'state_evidence_encoding': mode})
    if record.get('source_id') != view.get('id'):
        raise ValueError('Reader record must reference this observed source')
    if mode == 'hash':
        payload = {key: record[key] for key in ('start', 'stop', 'fields')}
        return hashed_token(json.dumps(payload, sort_keys=True))
    if set(ontology) != set(FIELD_ORDER):
        raise ValueError('Structured Reader state requires the frozen six-field ontology')
    capacity = view.get('reader_positions')
    if (isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)) or capacity <= 1):
        raise ValueError('Structured Reader state requires indexed source positions')
    capacity = int(capacity)-1  # Exclude the source metadata row from the reservation.
    values = []
    fields = record['fields']
    for name in FIELD_ORDER:
        categories = sorted(ontology[name])
        if 'unknown' not in categories or len(categories) != len(set(categories)):
            raise ValueError('Structured Reader ontology must contain unique categories including unknown')
    soft = (_soft_relations(record, view, ontology, speaker_normalization)
            if mode == 'structured_soft' else None)
    for name in FIELD_ORDER:
        categories = sorted(ontology[name])
        category = fields.get(name, 'unknown')
        if category not in categories:
            raise ValueError('Reader record has a category outside its frozen ontology')
        values.extend(soft[name].tolist() if soft is not None
                      else [float(value == category) for value in categories])
    relation = record.get('relation_confidence') or {}
    values.extend(_confidence(relation.get(name, 0.)) for name in FIELD_ORDER)
    diagnostic_keys = ('presence_confidence', 'span_confidence', 'confidence')
    values.extend(_confidence(record.get(name, 0.)) for name in diagnostic_keys)
    values.extend(_span(record, capacity))
    values.extend(_span(record.get('predicted_span', record), capacity))
    values.append(float(all(name in record for name in diagnostic_keys)))
    if len(values) > 63:
        raise ValueError('Reader ontology exceeds the 64-feature Controller evidence capacity')
    vector = np.zeros(64, dtype=np.float32)
    vector[:len(values)], vector[63] = values, 1.
    return vector


def strip_reader_rows(x, observed_count, evidence_count, mask=None):
    """Remove only intermediate Reader rows for a Controller diagnostic."""
    if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0
           for n in (observed_count, evidence_count)):
        raise ValueError('Observed/Reader row counts must be nonnegative integers')
    rows = np.asarray(x)
    stop = observed_count+evidence_count
    if rows.ndim != 2 or rows.shape[1] != 71 or stop > len(rows):
        raise ValueError('Reader row counts do not match the Controller state matrix')
    masked = None if mask is None else np.asarray(mask)
    if masked is not None and masked.shape != (len(rows),):
        raise ValueError('Candidate mask must align with the Controller state rows')
    if not evidence_count:
        return x, mask
    kept = np.concatenate((np.arange(observed_count), np.arange(stop, len(rows))))
    return rows[kept], None if masked is None else masked[kept]
