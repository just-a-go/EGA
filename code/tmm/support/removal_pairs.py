"""Natural, text-only citation deletion; controls are rule-defined."""
import json
import re
from pathlib import Path

import numpy as np

from .data import ObservationError, canonical, digest, seed_views, write_json


SYMPTOMS = r'\b(depress(?:ed|ion|ing)?|anxious|anxiety|sleep(?:ing)?|insomnia|hopeless(?:ness)?|suicid\w*|worthless(?:ness)?|appetite|panic)\b'
QUESTION = r'^(?:how|what|when|where|why|who|which|do|did|does|are|is|was|were|have|has|can|could|would|will|tell me)\b'
# Every pattern matches the complete normalized turn. No short acknowledgements.
TRANSACTION_RULES = [
    r'(?:hi|hello|good morning|good afternoon|good evening)(?: there)?',
    r'(?:goodbye|good bye|bye|bye bye|have a (?:nice|good|great) day)',
    r'(?:thank you|thanks)(?: very much| so much)? for (?:coming|your time|participating)(?: today)?',
    r'(?:hello|hi) (?:and )?thank you for (?:coming|participating)(?: today)?',
    r'(?:hi|hello) (?:i am|i m) ellie (?:thanks|thank you) for coming in(?: today)?',
    r'(?:we are|we re|we will be|we ll be) (?:starting|beginning) (?:the|this) interview(?: now)?',
    r'(?:this|the) interview (?:is|will be) recorded',
    r'(?:i am|i m) (?:going to|gonna) ask you (?:some|a few|a series of) questions',
    r'(?:that is|that s) (?:all|the end of) (?:the|my|our) questions(?: for today)?',
    r'(?:we are|we re) (?:done|finished) with (?:the|this) interview',
]
# Exact released transcript strings, including script labels; no claim of spoken delivery.
RELEASED_SCRIPTED_CONTROLS = [
    "IntroV4Confirmation (hi i'm ellie thanks for coming in today i was created to talk to "
    "people in a safe and secure environment i'm not a therapist but i'm here to learn "
    "about people and would love to learn about you i'll ask a few questions to get us "
    "started and please feel free to tell me anything your answers are totally confidential "
    "are you ok with this)",
    'bye (goodbye)',
    "hi i'm ellie thanks for coming in today i was created to talk to people in a safe and secure environment",
]
TRANSACTION_RULES.extend(re.escape(re.sub(r'[^a-z0-9]+', ' ', text.casefold()).strip())
                         for text in RELEASED_SCRIPTED_CONTROLS)
ADMIN_CLAIM = re.compile(r'\b(interview|interviewer|greet\w*|goodbye|farewell|hello|administrat\w*|'
                         r'procedur\w*|polite\w*|courtes\w*|cooperat\w*|rapport|interaction|'
                         r'participation|participating|participate[ds]?|thank\w*|recording|setup|appointment)\b', re.I)


def transaction_rule(text):
    """A conservative full-turn rule; punctuation is irrelevant to matching only."""
    normalized = re.sub(r'[^a-z0-9]+', ' ', text.casefold()).strip()
    return next((rule for rule in TRANSACTION_RULES if re.fullmatch(rule, normalized)), None)


def build_plan(record):
    plan = {'id': str(record['id']), 'participant': str(record['participant']),
            'views': [], 'candidates': [], 'controls': [], 'reason': None}
    try:
        text = canonical([v for v in record['views'] if v['modality'] == 'T'
                          and v.get('available', True) and v['stop'] > v['start']])
        initial = seed_views(dict(record, views=[v for v in text if v['stop']-v['start'] <= 20]), 'T')
        controls = [(v, transaction_rule(v['text'])) for v in text]
        controls = [(v, rule) for v, rule in controls if rule][:3]
        candidates = [v for v in text if v.get('speaker', '').lower() == 'participant'
                      and re.search(SYMPTOMS, v['text'], re.I)][:2]
        previous = []
        for candidate in candidates:
            before = [v for v in text if v['stop'] <= candidate['start']
                      and v.get('speaker', '').lower() != 'participant'
                      and ('?' in v['text'] or re.search(QUESTION, v['text'].strip(), re.I))]
            if before:
                previous.append(before[-1])
        plan['views'] = canonical(initial + [v for v, _ in controls] + candidates + previous)
        plan['candidates'] = [v['id'] for v in candidates]
        plan['controls'] = [{'source_id': v['id'], 'text': v['text'], 'rule': rule}
                            for v, rule in controls]
        if not candidates:
            plan['reason'] = 'no_prespecified_symptom_candidate'
        elif not controls:
            plan['reason'] = 'no_complete_transactional_turn'
    except ObservationError as exc:
        plan['reason'] = 'context_selection_failure: '+str(exc)
    return plan


def citation_query(plan, instruction):
    return (instruction + '\nProvide one source-grounded claim about the participant\'s '
            'depression-relevant experience, with one short exact continuous quotation, preferably '
            'at most six words. Preserve essential negation and temporal qualifiers; use a longer '
            'quotation if needed to preserve its meaning. '
            'Do not discuss greetings, interview participation, politeness or interview procedures. '
            'Use only one of these participant source IDs: '+json.dumps(plan['candidates'])+
            '. Return only JSON {"claim":"...","source_id":"...","quote":"..."}. '
            'Do not paraphrase the quotation. If unsupported, return empty strings.')


def parse_citation(response, plan):
    raw = response.get('answer', '').strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    try:
        citation, _ = json.JSONDecoder().raw_decode(raw)
    except (ValueError, TypeError):
        return None, 'invalid_citation_json'
    if not isinstance(citation, dict) or any(not isinstance(citation.get(k), str)
                                            or not citation[k].strip()
                                            for k in ('claim', 'source_id', 'quote')):
        return None, 'missing_citation_fields'
    if ADMIN_CLAIM.search(citation['claim']):
        return None, 'claim_overlaps_transactional_control_topic'
    if citation['source_id'] not in plan['candidates']:
        return None, 'citation_not_from_prespecified_participant_candidate'
    view = next(v for v in plan['views'] if v['id'] == citation['source_id'])
    quote = citation['quote']
    first = view['text'].find(quote)
    if first < 0:
        return None, 'quote_not_exact_source_substring'
    if view['text'].find(quote, first+1) >= 0:
        return None, 'ambiguous_quote_offsets'
    stop = first+len(quote)
    word = lambda char: bool(re.match(r'\w', char))
    if ((first > 0 and word(view['text'][first-1]) and word(quote[0])) or
            (stop < len(view['text']) and word(quote[-1]) and word(view['text'][stop]))):
        return None, 'quote_cuts_source_word'
    return dict(citation, start=first, stop=first+len(quote), original_text=view['text']), None


def choose_control(plan, citation, tokenizer):
    target = len(tokenizer.encode(citation['quote'], add_special_tokens=False))
    if target < 1:
        return None
    for control in plan['controls']:
        words = list(re.finditer(r'\S+', control['text']))
        for i, word in enumerate(words):
            for last in words[i:]:
                start, stop = word.start(), last.end()
                quote = control['text'][start:stop]
                if len(tokenizer.encode(quote, add_special_tokens=False)) == target:
                    return dict(control, quote=quote, start=start, stop=stop, token_count=target,
                                original_text=control['text'])
    return None


def delete_span(views, span):
    return [dict(v, text=v['text'][:span['start']]+v['text'][span['stop']:])
            if v['id'] == span['source_id'] else dict(v) for v in views]


def score(rows, cut, seed=27, repeats=2000):
    good = [row for row in rows if row.get('status') == 'evaluable']
    names = ['mean_abs_delta_p_cited', 'mean_abs_delta_p_control', 'paired_delta_difference',
             'flip_percent_cited', 'flip_percent_control', 'paired_flip_difference_pp']
    result = {'n_total': len(rows), 'n_evaluable': len(good),
              'coverage': len(good)/len(rows) if rows else 0., 'threshold': cut,
              'estimand': 'Conditional on a valid exact citation and token-matched rule-defined control',
              'control_status': 'Rule-defined task-irrelevant control',
              'citation_status': 'Model-generated claim with verified exact quote',
              'track': 'T; text-character deletion only; no audio altered or evaluated',
              'excluded': [{'id': r['id'], 'participant': r['participant'], 'reason': r.get('reason')}
                           for r in rows if r.get('status') != 'evaluable'],
              'bootstrap': {'unit': 'participant', 'repeats': repeats, 'seed': seed,
                            'interval': 'percentile 95%; conditional on evaluability'}}
    if not good:
        return result | {name: None for name in names} | {'ci95': {name: None for name in names}}
    groups = {}
    for row in good:
        baseline = row['full']['p']
        dc = abs(row['cited_deleted']['p']-baseline)
        di = abs(row['control_deleted']['p']-baseline)
        fc = 100.*((row['cited_deleted']['p'] >= cut) != (baseline >= cut))
        fi = 100.*((row['control_deleted']['p'] >= cut) != (baseline >= cut))
        groups.setdefault(row['participant'], []).append([dc, di, dc-di, fc, fi, fc-fi])
    matrix = np.asarray([np.mean(v, axis=0) for v in groups.values()])
    result.update(dict(zip(names, matrix.mean(axis=0).tolist())))
    rng = np.random.default_rng(seed)
    draws = np.asarray([matrix[rng.integers(len(matrix), size=len(matrix))].mean(axis=0)
                        for _ in range(repeats)])
    result['ci95'] = dict(zip(names, np.quantile(draws, [.025, .975], axis=0).T.tolist()))
    return result


def run(backbone, records, out, cut, seed=27):
    """Freeze plans before inference; resume each participant and each model call."""
    records = [r for r in records if r.get('split') == 'test']
    if len(records) != 47 or len({str(r['participant']) for r in records}) != 47:
        raise ValueError('Natural deletion diagnostic requires the 47 unique DAIC-WOZ test participants')
    if not np.isfinite(cut) or not 0 <= cut <= 1:
        raise ValueError('Supply the original development-selected classification threshold')
    if backbone.protocol.get('max_output_tokens', 2048) > 160:
        raise ValueError('Set this diagnostic protocol max_output_tokens to at most 160 before running')
    if backbone.protocol.get('max_input_tokens', 4096) > 4096:
        raise ValueError('The diagnostic context limit cannot exceed 4096 tokens')
    plans = [build_plan(r) for r in sorted(records, key=lambda r: str(r['id']))]
    declaration = {'schema': 'natural-text-removal-v1', 'plans': plans,
                   'instruction': backbone.instruction, 'protocol': backbone.protocol,
                   'model': backbone.spec, 'threshold': cut, 'seed': seed,
                   'rules': TRANSACTION_RULES, 'symptom_candidate_regex': SYMPTOMS,
                   'control_source_note': 'Released transcript text may include exact scripted prompt labels; '
                                          'these are not asserted to be verbatim naturally spoken utterances',
                   'released_scripted_controls': RELEASED_SCRIPTED_CONTROLS,
                   'quote_boundary_rule': 'Reject adjacent Unicode word characters across either quote boundary',
                   'preceding_question_regex': QUESTION,
                   'claim_exclusion_regex': ADMIN_CLAIM.pattern,
                   'implementation_sha256': digest(Path(__file__).read_text(encoding='utf-8')),
                   'no_human_labels_used': True, 'bootstrap_repeats': 2000}
    fingerprint = digest(declaration)
    destination = Path(out)/'removal_pairs.json'
    if destination.exists():
        output = json.loads(destination.read_text(encoding='utf-8'))
        if output['fingerprint'] != fingerprint:
            raise ValueError('Removal plan/model/instruction/protocol changed; use a new output directory')
    else:
        output = {'fingerprint': fingerprint, 'declaration': declaration, 'participants': []}
        write_json(destination, output)
    saved = {row['id']: row for row in output['participants']}
    if len(saved) != len(output['participants']):
        raise ValueError('Duplicate saved removal participant')
    for plan in plans:
        row = saved.get(plan['id'])
        if row is None:
            row = {'id': plan['id'], 'participant': plan['participant'], 'status': 'pending'}
            output['participants'].append(row)
        if row['status'] != 'pending':
            continue
        reason = plan['reason']
        if not reason:
            if 'citation_response' not in row:
                row['citation_response'] = backbone.score(
                    plan['views'], query=citation_query(plan, backbone.instruction), use_reader=False)
                write_json(destination, output)
            if row['citation_response']['status'] != 'ok':
                reason = 'citation_input_failure: '+str(row['citation_response'].get('error'))
            else:
                citation, reason = parse_citation(row['citation_response'], plan)
                if citation:
                    row['citation'] = citation
                    row['control'] = choose_control(plan, citation, backbone.tokenizer)
                    if row['control'] is None:
                        reason = 'no_equal_token_transactional_substring'
                    else:
                        row['citation']['token_count'] = row['control']['token_count']
        if not reason:
            write_json(destination, output)
            conditions = [('full', plan['views']),
                          ('cited_deleted', delete_span(plan['views'], row['citation'])),
                          ('control_deleted', delete_span(plan['views'], row['control']))]
            for name, views in conditions:
                if name not in row:
                    row[name] = backbone.score_unaugmented(views)
                    write_json(destination, output)
                if row[name]['status'] != 'ok':
                    reason = name+'_input_failure: '+str(row[name].get('error'))
                    break
        row.update(status='unevaluable' if reason else 'evaluable', reason=reason)
        write_json(destination, output)
        print(json.dumps({'probe': 'natural_text_removal', 'id': row['id'],
                          'status': row['status'], 'reason': reason}), flush=True)
    output['metrics'] = score(output['participants'], cut, seed)
    write_json(destination, output)
    return output['metrics']
