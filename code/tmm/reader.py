"""Eight-query evidence reader, source-linked decoding and presentation consistency."""
import json
import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from .support.data import canonical, digest
from .support.models import InputError
from .support.reader_inputs import (FIELDS, SourceEncoder, hashed_token, normalize_speaker,
                            normalize_relation_temperatures, SPEAKER_NORMALIZATION_MODES)


RELATION_CONDITIONING_MODES = ('query_only', 'predicted_span_mean')


class EvidenceReader(nn.Module):
    def __init__(self, ontology, width=128, relation_conditioning='query_only',
                 boundary_likelihood='legacy_independent'):
        super().__init__()
        if relation_conditioning not in RELATION_CONDITIONING_MODES:
            raise ValueError('Unsupported Reader relation_conditioning')
        if boundary_likelihood not in BOUNDARY_MODES:
            raise ValueError('Unsupported Reader boundary_likelihood')
        self.ontology = ontology
        self.relation_conditioning = relation_conditioning
        self.boundary_likelihood = boundary_likelihood
        self.project = nn.Linear(69, width)
        layer = nn.TransformerEncoderLayer(width, 4, 256, dropout=0., batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(8, width)*.02)
        self.attend = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.presence = nn.Linear(width, 1)
        self.start = nn.Linear(width, width)
        self.stop = nn.Linear(width, width)
        self.relations = nn.ModuleDict({key: nn.Linear(width, len(ontology[key])) for key in FIELDS})

    def encode_memory(self, sources, renderer=0):
        """Equivalent presentations keep pointer coordinates in unique source space."""
        if renderer not in (0, 1, 2):
            raise ValueError('Unknown reader presentation')
        rows, pointers = [], []
        for source in sources:
            view, values = source['view'], source['values']
            kind = [float(view['modality'] == m) for m in ('T', 'A', 'F')]
            meta = hashed_token(json.dumps({k: view.get(k) for k in ('speaker', 'modality', 'start', 'stop')}, sort_keys=True))
            metadata = [np.concatenate([meta, kind, [view['start']/3600, -1.]])]
            payload = [np.concatenate([value, kind, [view['start']/3600, j/max(1, len(values)-1)]])
                       for j, value in enumerate(values)]
            labels = [(view['id'], j) for j in range(len(values))]
            if renderer == 1:
                rows.extend(payload + metadata)
                pointers.extend(labels + [None])
            else:
                rows.extend(metadata + payload)
                pointers.extend([None] + labels)
        if renderer == 2:
            # Explicit duplicate marker has no source coordinate.
            source = sources[0]
            kind = [float(source['view']['modality'] == m) for m in ('T', 'A', 'F')]
            rows.append(np.concatenate([hashed_token('Repeated source; no new observation'), kind, [0., -1.]]))
            pointers.append(None)
            for j, value in enumerate(source['values']):
                rows.append(np.concatenate([value, kind, [source['view']['start']/3600, j/max(1,len(source['values'])-1)]]))
                pointers.append((source['view']['id'], j))
        device = self.queries.device
        hidden = self.project(torch.tensor(np.asarray(rows), dtype=torch.float32, device=device))
        positions = torch.arange(len(rows), device=device).float().unsqueeze(1)
        frequency = torch.exp(torch.arange(0, hidden.shape[1], 2, device=device)*(-math.log(10000.)/hidden.shape[1]))
        positional = torch.zeros_like(hidden)
        positional[:, 0::2], positional[:, 1::2] = torch.sin(positions*frequency), torch.cos(positions*frequency)
        memory = self.encoder((hidden+positional).unsqueeze(0))
        return memory, pointers

    def forward(self, sources, focal, renderer=0, encoded=None):
        memory, pointers = self.encode_memory(sources, renderer) if encoded is None else encoded
        source = next(s for s in sources if s['view']['id'] == focal)
        # Aggregate repeated positions BEFORE predicting boundaries on unique coordinates.
        unique = torch.stack([memory[0, [i for i, p in enumerate(pointers) if p == (focal,j)]].mean(0)
                              for j in range(len(source['values']))])
        query = (self.queries+unique.mean(0)).unsqueeze(0)
        attended, _ = self.attend(query, memory, memory, need_weights=False)
        queries = (query+attended)[0]
        output = {'presence': self.presence(queries).squeeze(-1),
                  'start': self.start(queries) @ unique.T / math.sqrt(unique.shape[-1]),
                  'stop': self.stop(queries) @ unique.T / math.sqrt(unique.shape[-1])}
        relation_queries = queries
        if self.relation_conditioning == 'predicted_span_mean':
            relation_queries = span_relation_features(queries, unique, output['start'], output['stop'],
                                                      source, self.boundary_likelihood)
        output['relations'] = {k: head(relation_queries) for k, head in self.relations.items()}
        return output


BOUNDARY_MODES = ('legacy_independent', 'ordered_conditional_stop_v1', 'global_legal_joint_v1')


def legal_span_mask(length, source=None, device=None):
    mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=device))
    if source is not None and source['view']['modality'] == 'F':
        names = [coordinate[0] for coordinate in source['coordinates']]
        if len(names) != length:
            raise ValueError('Reader feature coordinates do not match boundary logits')
        mask &= torch.tensor([[a == b for b in names] for a in names], device=device)
    return mask


def global_span_log_normalizer(start_logits, stop_logits, source=None):
    """One global legal-joint partition function, without allocating Q x N x N."""
    if start_logits.shape != stop_logits.shape or not start_logits.shape[-1]:
        raise ValueError('Reader start/stop logits must have the same nonempty shape')
    if source is None or source['view']['modality'] != 'F':
        suffix = torch.logcumsumexp(stop_logits.flip(-1), -1).flip(-1)
        return torch.logsumexp(start_logits+suffix, -1)
    names = [coordinate[0] for coordinate in source['coordinates']]
    if len(names) != start_logits.shape[-1]:
        raise ValueError('Reader feature coordinates do not match boundary logits')
    components = []
    for name in dict.fromkeys(names):
        indices = [index for index, category in enumerate(names) if category == name]
        stop = stop_logits[..., indices]
        suffix = torch.logcumsumexp(stop.flip(-1), -1).flip(-1)
        components.append(start_logits[..., indices]+suffix)
    return torch.logsumexp(torch.cat(components, -1), -1)


def global_gold_log_probs(start_logits, stop_logits, first, last, source=None):
    if not 0 <= first <= last < start_logits.shape[-1]:
        raise ValueError('Invalid Reader supervised span')
    if source is not None and source['view']['modality'] == 'F':
        if source['coordinates'][first][0] != source['coordinates'][last][0]:
            raise ValueError('Reader supervised span crosses feature coordinates')
    return (start_logits[..., first]+stop_logits[..., last]
            -global_span_log_normalizer(start_logits, stop_logits, source))


def ordered_stop_log_probs(logits, first):
    """The supervised end distribution, conditional on a legal start position."""
    positions = torch.arange(logits.shape[-1], device=logits.device)
    return logits.masked_fill(positions < first, -torch.inf).log_softmax(-1)


def span_log_probs(start_logits, stop_logits, boundary_likelihood='legacy_independent', source=None):
    if boundary_likelihood not in BOUNDARY_MODES:
        raise ValueError('Unsupported Reader boundary_likelihood')
    if boundary_likelihood == 'global_legal_joint_v1':
        joint = start_logits[:, None]+stop_logits[None, :]
        return (joint-global_span_log_normalizer(start_logits, stop_logits, source)).masked_fill(
            ~legal_span_mask(len(start_logits), source, start_logits.device), -torch.inf)
    start = start_logits.log_softmax(-1)
    if boundary_likelihood == 'legacy_independent':
        return start[:, None]+stop_logits.log_softmax(-1)[None, :]
    # logsumexp of each legal suffix avoids a second N x N softmax tensor.
    normalizer = torch.logcumsumexp(stop_logits.flip(-1), -1).flip(-1)
    joint = start[:, None]+stop_logits[None, :]-normalizer[:, None]
    return joint.masked_fill(~torch.triu(torch.ones_like(joint, dtype=torch.bool)), -torch.inf)


def select_predicted_span(start_logits, stop_logits, source, boundary_likelihood='legacy_independent',
                          joint_log_probs=None):
    """Select one legal original source span; stop is exclusive."""
    joint = (span_log_probs(start_logits, stop_logits, boundary_likelihood, source=source)
             if joint_log_probs is None else joint_log_probs)
    mask = legal_span_mask(len(start_logits), source, joint.device)
    index = int(joint.masked_fill(~mask, -torch.inf).argmax())
    first, last = divmod(index, stop_logits.shape[-1])
    return first, last+1


def span_relation_features(queries, unique, start_logits, stop_logits, source,
                           boundary_likelihood='legacy_independent'):
    """Condition relations on predicted source hidden states, never gold/context spans."""
    spans = [select_predicted_span(start, stop, source, boundary_likelihood)
             for start, stop in zip(start_logits, stop_logits)]
    means = torch.stack([unique[first:stop].mean(0) for first, stop in spans])
    return queries+means


def matching(output, targets, ontology, boundary_likelihood='legacy_independent', source=None):
    if not targets:
        return [], []
    cost = []
    for target in targets:
        value = -F.logsigmoid(output['presence'])
        if boundary_likelihood == 'global_legal_joint_v1':
            value = value-global_gold_log_probs(output['start'], output['stop'],
                target['start'], target['stop']-1, source)
        else:
            value = value-output['start'].log_softmax(-1)[:, target['start']]
            stop = (ordered_stop_log_probs(output['stop'], target['start'])
                    if boundary_likelihood == 'ordered_conditional_stop_v1'
                    else output['stop'].log_softmax(-1))
            value = value-stop[:, target['stop']-1]
        for key, category in target['fields'].items():
            value = value-output['relations'][key].log_softmax(-1)[:, ontology[key].index(category)]
        cost.append(value.detach().cpu().numpy())
    queries, labels = linear_sum_assignment(np.stack(cost, axis=1))
    return queries.tolist(), labels.tolist()


def human_loss(output, annotation, ontology, class_weights=None,
               boundary_likelihood='legacy_independent', source=None):
    targets = annotation['records']
    queries, labels = matching(output, targets, ontology, boundary_likelihood, source)
    zero = output['presence'].sum()*0.
    sums, counts = {}, {}
    active = list(range(8)) if annotation.get('annotation_complete', False) else queries
    if active:
        desired = torch.zeros_like(output['presence'])
        desired[queries] = 1
        sums['presence'] = F.binary_cross_entropy_with_logits(output['presence'][active], desired[active], reduction='sum')
        counts['presence'] = len(active)
    for query, label in zip(queries, labels):
        target = targets[label]
        for name, index in (('start', target['start']), ('stop', target['stop']-1)):
            # Enforce ordered end boundaries during supervised training too.
            logits = output[name][query]
            if boundary_likelihood == 'global_legal_joint_v1':
                # Existing unit start/stop weights together count one joint NLL.
                value = -.5*global_gold_log_probs(output['start'][query], output['stop'][query],
                    target['start'], target['stop']-1, source)
            elif name == 'stop':
                value = -ordered_stop_log_probs(logits, target['start'])[index]
            else:
                value = -logits.log_softmax(-1)[index]
            sums[name], counts[name] = sums.get(name, zero)+value, counts.get(name, 0)+1
        for name, category in target['fields'].items():
            category_index = ontology[name].index(category)
            value = -output['relations'][name][query].log_softmax(-1)[category_index]
            if class_weights is not None:
                value = value*class_weights[name][category_index]
            sums[name], counts[name] = sums.get(name, zero)+value, counts.get(name, 0)+1
    return sums, counts, (queries, labels)


def js_divergence(left, right):
    left, right = left.clamp_min(1e-8), right.clamp_min(1e-8)
    middle = (left+right)/2
    return .5*((left*(left.log()-middle.log())).sum(-1)+(right*(right.log()-middle.log())).sum(-1))


def consistency_loss(left, right, left_match, right_match, annotation_complete,
                     boundary_likelihood='legacy_independent', source=None):
    joint = boundary_likelihood == 'global_legal_joint_v1'
    if joint:
        # Exclude masked -inf entries before taking logs/KL; both renderers share source coordinates.
        first, last = legal_span_mask(left['start'].shape[-1], source, left['start'].device).nonzero(as_tuple=True)
        def probabilities(output):
            normalizer = global_span_log_normalizer(output['start'], output['stop'], source)
            return (output['start'][:, first]+output['stop'][:, last]-normalizer[:, None]).exp()
        left_span, right_span = probabilities(left), probabilities(right)
    q0, labels0 = left_match
    qr, labelsr = right_match
    pairs = [(q0[labels0.index(label)], qr[labelsr.index(label)]) for label in labels0]
    if annotation_complete:
        # Align remaining set queries by presence and boundary distributions.
        a, b = [q for q in range(8) if q not in q0], [q for q in range(8) if q not in qr]
        if a:
            with torch.no_grad():
                if joint:
                    cost = torch.stack([torch.stack([js_divergence(left_span[i], right_span[j])
                                                   for j in b]) for i in a])
                else:
                    cost = js_divergence(left['start'][a].softmax(-1)[:, None], right['start'][b].softmax(-1)[None, :])
                ii, jj = linear_sum_assignment(cost.cpu().numpy())
            pairs.extend((a[i], b[j]) for i, j in zip(ii, jj))
    if not pairs:
        return left['presence'].sum()*0.
    a, b = zip(*pairs)
    p, q = left['presence'][list(a)].sigmoid(), right['presence'][list(b)].sigmoid()
    losses = [js_divergence(torch.stack([p,1-p], -1), torch.stack([q,1-q], -1)).mean()]
    if joint:
        span_loss = js_divergence(left_span[list(a)], right_span[list(b)]).mean()
        # Keep the existing two boundary slots' consistency weight, using the same joint in both.
        losses.extend((span_loss, span_loss))
    else:
        for key in ('start', 'stop'):
            losses.append(js_divergence(left[key][list(a)].softmax(-1), right[key][list(b)].softmax(-1)).mean())
    for key in FIELDS:
        losses.append(js_divergence(left['relations'][key][list(a)].softmax(-1), right['relations'][key][list(b)].softmax(-1)).mean())
    return torch.stack(losses).mean()


def _sentence_context(source, first, last):
    """Extend a token span within the same source, retaining sentence context."""
    offsets = source.get('offsets')
    if offsets is None:
        return 0, len(source['coordinates'])-1, source['view'].get('text', ''.join(source['excerpts']))
    text = source['view']['text']
    begin = min(a for a, b in offsets[first:last+1])
    end = max(b for a, b in offsets[first:last+1])
    boundaries = '.!?。！？\n'
    begin = max(text.rfind(mark, 0, begin)+1 for mark in boundaries)
    if not end or text[end-1] not in boundaries:
        stops = [text.find(mark, end) for mark in boundaries]
        end = min((stop+1 for stop in stops if stop >= 0), default=len(text))
    positions = [i for i, (a, b) in enumerate(offsets) if b > begin and a < end]
    return positions[0], positions[-1], text[begin:end]


def _predicted_span(record):
    span = record.get('predicted_span', record)
    return span['start'], span['stop']


def _merge_overlapping(records):
    """Suppress near-identical queries, not independent claims sharing context."""
    kept = []
    for record in sorted(records, key=lambda r: -r['confidence']):
        first, stop = _predicted_span(record)
        duplicate = False
        for other in kept:
            if record['source_id'] != other['source_id']:
                continue
            a, b = _predicted_span(other)
            overlap = max(0, min(stop, b)-max(first, a))
            union = stop-first+b-a-overlap
            if overlap/union >= .8:
                duplicate = True
                break
        if not duplicate:
            kept.append(record)
    return sorted(kept, key=lambda r: (r['source_id'], *_predicted_span(r)))


def decode(output, source, ontology, threshold, supported, relation_threshold=.7, record_threshold=0.,
           boundary_likelihood='legacy_independent', include_relation_probabilities=False,
           speaker_normalization='literal', relation_temperatures=None):
    """Abstain on uncertain fields and keep one query per unique source span."""
    temperatures = normalize_relation_temperatures(relation_temperatures)
    records = {}
    for query in range(8):
        presence_confidence = float(output['presence'][query].sigmoid())
        if presence_confidence < threshold:
            continue
        joint = span_log_probs(output['start'][query], output['stop'][query], boundary_likelihood, source=source)
        first, stop = select_predicted_span(output['start'][query], output['stop'][query],
                                           source, boundary_likelihood, joint_log_probs=joint)
        last = stop-1
        fields, relation_confidence, relation_probabilities = {}, {}, {}
        for key in FIELDS:
            logits = output['relations'][key][query]
            probability = ((logits/temperatures[key]).softmax(-1) if key in temperatures
                           else logits.softmax(-1))
            category = int(probability.argmax())
            confidence = float(probability[category]) if key in supported else 0.
            relation_confidence[key] = confidence
            fields[key] = ontology[key][category] if confidence >= relation_threshold else 'unknown'
            if include_relation_probabilities:
                relation_probabilities[key] = {
                    name: float(probability[index]) if key in supported else float(name == 'unknown')
                    for index, name in enumerate(ontology[key])}
        # Known source identity is reliable metadata; unknown identity stays unknown.
        fields['speaker'] = normalize_speaker(source['view'].get('speaker', 'unknown'), ontology,
                                              speaker_normalization)
        relation_confidence['speaker'] = float(fields['speaker'] != 'unknown')
        if include_relation_probabilities:
            relation_probabilities['speaker'] = {
                name: float(name == fields['speaker']) for name in ontology['speaker']}
        span_confidence = float(joint[first, last].exp().sqrt())
        learned_confidence = [relation_confidence[key] for key in supported if key != 'speaker']
        confidence = presence_confidence*span_confidence*(float(np.mean(learned_confidence)) if learned_confidence else 1.)
        record = {'source_id': source['view']['id'], 'modality': source['view']['modality'],
                  'start': first, 'stop': last+1, 'fields': fields,
                  'source_start': source['coordinates'][first], 'source_stop': source['coordinates'][last],
                  'presence_confidence': round(presence_confidence, 6), 'span_confidence': round(span_confidence, 6),
                  'relation_confidence': {key: round(value, 6) for key, value in relation_confidence.items()},
                  'confidence': round(confidence, 6)}
        if include_relation_probabilities:
            record.update(relation_probabilities=relation_probabilities,
                          relation_probability_basis='reader_relation_softmax_v1',
                          relation_probability_supported=sorted(supported))
        if source['view']['modality'] == 'T':
            offsets = source.get('offsets')
            if offsets is not None:
                begin = min(a for a, b in offsets[first:last+1])
                end = max(b for a, b in offsets[first:last+1])
                record['text'] = source['view']['text'][begin:end]
            else:
                record['text'] = ''.join(source['excerpts'][first:last+1])
            # Keep short meaningful answers/denials, such as 'no' or '不', intact.
            if not any(char.isalnum() for char in record['text']):
                continue
            record['predicted_span'] = {key: record[key] for key in
                                        ('start', 'stop', 'source_start', 'source_stop')}
            if include_relation_probabilities:
                record['predicted_span_text'] = record['text']
            context_first, context_last, context = _sentence_context(source, first, last)
            record.update(start=context_first, stop=context_last+1, text=context,
                          source_start=source['coordinates'][context_first],
                          source_stop=source['coordinates'][context_last])
        key = record['source_id'], first, last+1
        if key not in records or record['confidence'] > records[key]['confidence']:
            records[key] = record
    return [record for record in _merge_overlapping(list(records.values()))
            if record['confidence'] >= record_threshold]


class ReaderRuntime:
    runtime_contract = 'runtime_selected_capacity_growth_v2'

    def __init__(self, backbone, bundle, include_relation_probabilities=False):
        if bundle['fingerprint'] != reader_fingerprint(bundle):
            raise ValueError('Reader checkpoint fingerprint mismatch')
        expected = bundle.get('backbone_spec')
        if expected is not None and expected != getattr(backbone, 'spec', {}):
            raise ValueError('Reader checkpoint requires its original frozen backbone representation')
        self.config, self.bundle = bundle['config'], bundle
        self.relation_temperatures = normalize_relation_temperatures(self.config.get('relation_temperatures'))
        decoder_temperatures = normalize_relation_temperatures(
            bundle.get('decoder_policy', {}).get('relation_temperatures'))
        if self.relation_temperatures != decoder_temperatures:
            raise ValueError('Reader relation temperatures config/decoder policy mismatch')
        self.encoder = SourceEncoder(backbone, self.config)
        boundary_likelihood = bundle.get('decoder_policy', {}).get('boundary_likelihood',
            self.config.get('boundary_likelihood', 'legacy_independent'))
        self.speaker_normalization = bundle.get('decoder_policy', {}).get('speaker_normalization',
            self.config.get('speaker_normalization', 'literal'))
        if self.speaker_normalization not in SPEAKER_NORMALIZATION_MODES:
            raise ValueError('Unsupported Reader speaker_normalization')
        if ('global_legal_joint_v1' in (boundary_likelihood, self.config.get('boundary_likelihood')) and
                boundary_likelihood != self.config.get('boundary_likelihood', 'legacy_independent')):
            raise ValueError('Reader boundary config/decoder policy mismatch')
        if ('ontology_casefold_v1' in (self.speaker_normalization, self.config.get('speaker_normalization')) and
                self.speaker_normalization != self.config.get('speaker_normalization', 'literal')):
            raise ValueError('Reader speaker config/decoder policy mismatch')
        self.model = EvidenceReader(self.config['ontology'],
                                   relation_conditioning=self.config.get('relation_conditioning', 'query_only'),
                                   boundary_likelihood=boundary_likelihood)
        self.model.load_state_dict(bundle['state_dict'])
        self.model.eval().requires_grad_(False)
        self.fingerprint = bundle['fingerprint']
        self.include_relation_probabilities = include_relation_probabilities

    def selected_views(self, views):
        policy = self.bundle.get('decoder_policy', {}).get('memory_modalities',
                                                          self.config.get('memory_modalities', 'all_observed'))
        if policy not in ('all_observed', 'supervised'):
            raise ValueError('Unsupported Reader memory_modalities')
        allowed = self.bundle.get('supervised_modalities', list(self.bundle['supported']))
        return [v for v in canonical(views) if policy == 'all_observed' or v['modality'] in allowed]

    def reserved_positions(self, views):
        # Indexed T reservations remain the UTF-8 byte upper bound; only the
        # observations this Runtime actually encodes consume Reader positions.
        selected = self.selected_views(views)
        if any('reader_positions' not in view for view in selected):
            raise ValueError('Selected Reader observation lacks an indexed position reservation')
        return sum(view['reader_positions'] for view in selected)

    def candidate_capacity_exceeded(self, views, action):
        # An unselected candidate leaves Reader memory unchanged. Existing T's
        # byte bound can exceed its already-validated actual encoded length.
        return bool(self.selected_views([action])) and (
            self.reserved_positions(views+[action]) > self.config['max_positions'])

    def sources(self, views):
        sources = [self.encoder.encode(v) for v in self.selected_views(views)]
        if sum(len(s['values'])+1 for s in sources) > self.config['max_positions']:
            raise InputError('Evidence reader context exceeds position limit; no silent truncation')
        return sources

    def read(self, views):
        sources = self.sources(views)
        if not sources:
            return []
        records = []
        include_relation_probabilities = getattr(self, 'include_relation_probabilities', False)
        with torch.no_grad():
            encoded = self.model.encode_memory(sources)
            for source in sources:
                output = self.model(sources, source['view']['id'], encoded=encoded)
                found = decode(output, source, self.config['ontology'], self.bundle['threshold'],
                               self.bundle['supported'].get(source['view']['modality'], []),
                                self.config.get('relation_confidence_threshold', .7),
                                self.bundle.get('record_confidence_threshold', 0.),
                                self.bundle.get('decoder_policy', {}).get('boundary_likelihood',
                                    self.config.get('boundary_likelihood', 'legacy_independent')),
                                include_relation_probabilities=include_relation_probabilities,
                                speaker_normalization=self.speaker_normalization,
                                relation_temperatures=self.relation_temperatures)
                if source['view']['modality'] == 'T' and (source.get('offsets') is None or
                                                         include_relation_probabilities):
                    tokens = self.encoder.backbone.tokenizer.encode(source['view']['text'], add_special_tokens=False)
                    for record in found:
                        if source.get('offsets') is None:
                            record['text'] = self.encoder.backbone.tokenizer.decode(tokens[record['start']:record['stop']])
                        if include_relation_probabilities:
                            span = record['predicted_span']
                            record['predicted_span_text'] = self.encoder.backbone.tokenizer.decode(
                                tokens[span['start']:span['stop']])
                records.extend(found)
        return records


def reader_fingerprint(bundle):
    weights = {key: value.detach().cpu().tolist() for key, value in bundle['state_dict'].items()}
    return digest({k: v for k, v in bundle.items() if k not in ('state_dict', 'fingerprint')} | {'weights': weights})
