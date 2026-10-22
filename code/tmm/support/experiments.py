"""EGA training and paired evidence interventions."""
import argparse
import gc
import hashlib
import io
import json
import os
import time
from pathlib import Path

import torch

from .data import ObservationError, canonical, digest, load_records, pool, read_jsonl, seed_views, write_json
from ..controller import (COST_BASIS, Controller, STATE_REPRESENTATION_VERSION, acquire, fit, fit_cost,
                  reader_gate_signature, select_policy, teacher_pairs)
from .features import fit_adapter
from .metrics import bootstrap, classification, paired, paired_bootstrap, threshold
from .models import Backbone
from .reader_training import fit_reader, index_reader_positions
from ..reader import ReaderRuntime
from .reader_inputs import load_annotations, validate_config, annotations_for
from . import grpo


def row(rec, prediction, cut=None):
    value = {k: rec[k] for k in ('id', 'participant', 'family', 'label')}
    value.update(prediction)
    if cut is not None:
        value['threshold'] = cut
    return value


def seed_predictions(backbone, records, track, panel=False):
    results = []
    for rec in records:
        try:
            score = getattr(backbone, 'score_unaugmented', backbone.score)
            prediction = score(seed_views(rec, track, panel))
        except ObservationError as exc:
            prediction = {'p': .5, 'status': 'input_failure', 'error': str(exc), 'seconds': 0., 'gpu_seconds': 0.}
        results.append(row(rec, prediction))
    return results


def _record_signature(records):
    return [{**{k: v for k, v in rec.items() if k not in ('index_seconds', 'views')},
             'views': [{k: v for k, v in view.items() if k != 'reader_positions'}
                       for view in rec['views']]} for rec in records]


def _discriminator_fingerprint(backbone):
    bundle = getattr(backbone, 'discriminator_bundle', None)
    return bundle['fingerprint'] if bundle else None


def _mark_discriminator_targets(examples, backbone):
    fingerprint = _discriminator_fingerprint(backbone)
    for example in examples:
        example['discriminator_fingerprint'] = fingerprint


def _cached_teacher_pairs(backbone, records, track, config, directory, resume):
    """Persist expensive frozen-model targets after each complete source record."""
    examples, timings = [], []
    for index, rec in enumerate(records):
        key = digest((rec['family'], rec['id']))[:20]
        path = directory / (key+'.pt')
        expected = grpo.signature((backbone.reader.fingerprint, _discriminator_fingerprint(backbone),
                                   track, config, _record_signature([rec])))
        cached = grpo.load_checkpoint(path, resume, 'ega-teacher-record-v1', expected)
        if cached is None:
            record_config = dict(config, seed=(int(config['seed'])+int(key[:8], 16)) % (2**32))
            excluded = None
            try:
                pairs, measured = teacher_pairs(backbone, [rec], track, record_config)
            except ObservationError as exc:
                pairs, measured, excluded = [], [], str(exc)
            except ValueError as exc:
                if str(exc) != 'No eligible teacher actions; inspect source eligibility and interfaces':
                    raise
                pairs, measured, excluded = [], [], str(exc)
            _mark_discriminator_targets(pairs, backbone)
            cached = {'schema': 'ega-teacher-record-v1', 'signature': expected,
                      'pairs': pairs, 'timings': measured, 'excluded': excluded}
            grpo.save_checkpoint(path, cached)
        examples.extend(cached['pairs'])
        timings.extend(cached['timings'])
        print(json.dumps({'teacher_record': rec['id'], 'done': index+1, 'total': len(records),
                          'pairs': len(cached['pairs']), 'excluded': cached['excluded']}), flush=True)
    return examples, timings


def _validate_plain_teacher_origin(backbone, train, dev, track, config, origin):
    """Permit Reader replacement, never a changed native scorer or source cohort."""
    identity = {'backbone_spec': backbone.spec, 'protocol': config.get('protocol', {}),
                'instruction': backbone.instruction, 'discriminator_fingerprint': _discriminator_fingerprint(backbone),
                'track': track, 'renderers': [0, 1, 2]}
    if (origin.get('schema') != 'ega-plain-teacher-origin-v1' or origin.get('native_identity') != identity
            or origin.get('source_records_signature') != grpo.signature((_record_signature(train), _record_signature(dev)))
            or not all(origin.get(key) for key in ('checkpoint', 'checkpoint_sha256', 'fit_signature',
                                                  'fit_protocol_sha256', 'source_reader_fingerprint', 'raw_files'))):
        raise ValueError('Plain Teacher origin changed its native scorer or source identity')
    old = teacher_generation_config(origin['source_teacher_config'])
    new = teacher_generation_config(config['controller'])
    if (old.get('reader_gate') != origin.get('source_reader_gate')
            or {k: v for k, v in old.items() if k != 'reader_gate'}
               != {k: v for k, v in new.items() if k != 'reader_gate'}):
        raise ValueError('Plain Teacher generation or renderer contract changed')
    if not config.get('reader_refit', False):
        from .reader_state import reader_state_mode
        if (reader_state_mode(config['controller']) != 'structured_soft'
                or not getattr(backbone.reader, 'include_relation_probabilities', False)):
            raise ValueError('Unchanged Reader plain Teacher reuse requires its actual soft probability payload')


def _refit_plain_teacher_pairs(backbone, train, dev, track, config, origin, directory, resume):
    """Read immutable native scores and derive new Reader states, or score a fresh record."""
    from .reader_teacher_refit import derive_refit_reader_teacher
    from .reader_state import reader_state_mode, reader_state_profile
    merged, train_timings, files, counts = [], [], {}, {}
    source = Path(origin['checkpoint']).parent
    for split, records in (('train', train), ('dev', dev)):
        examples = []
        counts[split] = {'derived': 0, 'fresh': 0}
        for rec in records:
            key = digest((rec['family'], rec['id']))[:20]
            relative = f'teacher_raw/{split}/{key}.pt'
            destination = directory/split/(key+'.pt')
            record_signature = _record_signature([rec])
            expected = grpo.signature((backbone.reader.fingerprint, _discriminator_fingerprint(backbone),
                                       track, config, record_signature))
            derivation_identity = ('ega-refit-plain-teacher-v1', origin, expected)
            if reader_state_mode(config) == 'structured_soft':
                derivation_identity += (reader_state_profile(config),)
            derivation = grpo.signature(derivation_identity)
            cached = grpo.load_checkpoint(destination, resume, 'ega-teacher-record-v1', expected)
            if cached is not None and cached.get('reader_refit_signature') != derivation:
                raise ValueError('Derived plain Teacher cache belongs to another source or Reader')
            if cached is None:
                pairs, measured = None, []
                report = {'reusable': False, 'reason': 'reader_gate_accepted_or_source_record_unavailable'}
                raw_path, raw_sha = source/relative, origin['raw_files'].get(relative)
                if ((origin.get('source_reader_gate') or {}).get('accept') is False
                        and (config.get('reader_gate') or {}).get('accept') is False
                        and raw_sha and raw_path.is_file()):
                    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != raw_sha:
                        raise ValueError('Original plain Teacher raw file changed')
                    old_signature = grpo.signature((origin['source_reader_fingerprint'],
                        _discriminator_fingerprint(backbone), track, origin['source_teacher_config'], record_signature))
                    raw = grpo.load_checkpoint(raw_path, True, 'ega-teacher-record-v1', old_signature)
                    verified = dict(origin, record_signature=grpo.signature(record_signature),
                                    source_record_cache_sha256=raw_sha,
                                    destination_reader_gate=config.get('reader_gate'))
                    pairs, measured, report = derive_refit_reader_teacher(
                        backbone, rec, raw['pairs'], raw['timings'], config, verified)
                derived = pairs is not None
                if derived:
                    _mark_discriminator_targets(pairs, backbone)
                    cached = {'schema': 'ega-teacher-record-v1', 'signature': expected,
                              'pairs': pairs, 'timings': measured, 'excluded': None}
                else:
                    _cached_teacher_pairs(backbone, [rec], track, config, directory/split, False)
                    cached = torch.load(destination, map_location='cpu', weights_only=False)
                cached.update(reader_refit_signature=derivation, reader_refit_derivation=report,
                              fixed_source_reference_cost_profile=derived)
                grpo.save_checkpoint(destination, cached)
            counts[split]['derived' if cached['fixed_source_reference_cost_profile'] else 'fresh'] += 1
            examples.extend(cached['pairs'])
            if split == 'train':
                train_timings.extend(cached['timings'])
            files[relative] = hashlib.sha256(destination.read_bytes()).hexdigest()
            print('reader_refit_teacher: '+json.dumps({'split': split, 'record': rec['id'],
                  'derived': cached['fixed_source_reference_cost_profile'], 'pairs': len(cached['pairs'])}), flush=True)
        merged.append(examples)
    provenance = {'mode': 'source_plain_teacher_reader_refit_v1',
                  'cost_profile': 'fixed_source_reference_cost_profile',
                  'cost_scope': 'derived_records_only; fresh records retain their measured costs',
                  'new_reader_seconds': 'diagnostic_only; online acquisition uses actual wallclock',
                  'base_checkpoint_sha256': origin['checkpoint_sha256'],
                  'base_fit_signature': origin['fit_signature'], 'base_fit_protocol_sha256': origin['fit_protocol_sha256'],
                  'base_raw_files': origin['raw_files'], 'derived_files': files, 'counts': counts,
                  'source_reader_fingerprint': origin['source_reader_fingerprint'],
                  'reader_fingerprint': backbone.reader.fingerprint,
                  'discriminator_fingerprint': _discriminator_fingerprint(backbone),
                  'source_records_signature': origin['source_records_signature']}
    if getattr(backbone.reader, 'include_relation_probabilities', False):
        provenance['reader_payload_policy'] = {'include_relation_probabilities': True}
    return (merged[0], merged[1], train_timings), provenance


def _cached_seed_predictions(backbone, records, track, path, resume):
    expected = grpo.signature((backbone.spec, _discriminator_fingerprint(backbone),
                               track, _record_signature(records)))
    cached = grpo.load_checkpoint(path, resume, 'ega-calibration-v1', expected)
    values = cached['predictions'] if cached else []
    done = {(r['family'], r['id']) for r in values}
    for rec in records:
        if (rec['family'], rec['id']) in done:
            continue
        values.extend(seed_predictions(backbone, [rec], track))
        grpo.save_checkpoint(path, {'schema': 'ega-calibration-v1', 'signature': expected,
                                    'predictions': values})
    return values


def _cached_reader_predictions(backbone, records, track, path, resume):
    expected = grpo.signature((backbone.spec, backbone.reader.fingerprint,
                               _discriminator_fingerprint(backbone), track, _record_signature(records)))
    cached = grpo.load_checkpoint(path, resume, 'ega-reader-calibration-v1', expected) if path else None
    values = cached['predictions'] if cached else []
    done = {(r['family'], r['id']) for r in values}
    for rec in records:
        if (rec['family'], rec['id']) in done:
            continue
        values.append(row(rec, backbone.score(seed_views(rec, track))))
        if path:
            grpo.save_checkpoint(path, {'schema': 'ega-reader-calibration-v1', 'signature': expected,
                                        'predictions': values})
    return values


def fit_reader_gate(plain, enhanced, minimum_gain=0., gain_loss=None):
    """Source development selects whether evidence is useful; target labels never enter."""
    before, after = {r['id']: r for r in plain}, {r['id']: r for r in enhanced}
    if len(before) != len(plain) or len(after) != len(enhanced) or before.keys() != after.keys():
        raise ValueError('Reader gate requires unique matched source development predictions')
    if any(before[key]['label'] != after[key]['label'] for key in before):
        raise ValueError('Reader gate labels changed within a source-development pair')
    valid = [key for key in before if all(r['status'] == 'ok' and r.get('probability_valid', True)
                                        for r in (before[key], after[key]))]
    if not valid:
        raise ValueError('No valid source-development pairs for Reader acceptance')
    plain_brier = sum((before[key]['p']-before[key]['label'])**2 for key in valid)/len(valid)
    enhanced_brier = sum((after[key]['p']-before[key]['label'])**2 for key in valid)/len(valid)
    from ..controller import gain_loss_settings, gain_probability
    profile = gain_loss_settings({'gain_loss': gain_loss or {}})
    plain_loss, reader_loss = plain_brier, enhanced_brier
    if profile['mode'] != 'raw_brier':
        left = gain_probability([before[key]['p'] for key in valid], profile)
        right = gain_probability([after[key]['p'] for key in valid], profile)
        labels = [before[key]['label'] for key in valid]
        plain_loss = float(sum((p-y)**2 for p, y in zip(left, labels))/len(valid))
        reader_loss = float(sum((p-y)**2 for p, y in zip(right, labels))/len(valid))
    gain = plain_loss-reader_loss
    gate = {'version': 1, 'accept': gain > minimum_gain, 'selection': 'source_dev_brier',
            'source_n': len(before), 'valid_pairs': len(valid), 'plain_brier': plain_brier,
            'reader_brier': enhanced_brier, 'brier_gain': plain_brier-enhanced_brier, 'minimum_gain': minimum_gain}
    if profile['mode'] != 'raw_brier':
        gate.update(version=2, selection='source_dev_controller_gain_loss', gain_loss=profile,
                    plain_loss=plain_loss, reader_loss=reader_loss, loss_gain=gain)
    gate['fingerprint'] = digest(gate)
    return gate


def joint_threshold(plain, enhanced):
    """One cutoff, selected equally across the two source-development conditions."""
    before, after = {r['id']: r for r in plain}, {r['id']: r for r in enhanced}
    if len(before) != len(plain) or len(after) != len(enhanced) or before.keys() != after.keys():
        raise ValueError('Joint calibration requires unique matched source development predictions')
    if any(before[key]['label'] != after[key]['label'] for key in before):
        raise ValueError('Joint calibration labels changed within a development pair')
    valid = [[{k: v for k, v in r.items() if k != 'threshold'} for r in rows
              if r['status'] == 'ok' and r.get('probability_valid', True)] for rows in (plain, enhanced)]
    if any(not rows for rows in valid):
        raise ValueError('Both conditions require valid source-development predictions')
    candidates = sorted({0., .5, 1.0000001, *[r['p'] for rows in valid for r in rows]})
    cut = max(candidates, key=lambda t: (sum(classification(rows, t)['macro_f1'] for rows in valid)/2,
                                         -abs(t-.5), -t))
    return cut, {'selection': 'joint_source_dev_macro_f1', 'target_adaptation': False,
                 'seed': classification(valid[0], cut), 'ega': classification(valid[1], cut)}


def shared_source_threshold(plain, enhanced, objective='mean_seed_ega_macro_f1'):
    """Keep the Seed cutoff independent of the acquisition policy when requested."""
    if objective == 'mean_seed_ega_macro_f1':
        return joint_threshold(plain, enhanced)
    if objective != 'seed_shared_source_dev':
        raise ValueError('Unsupported source-development threshold objective')
    before, after = {r['id']: r for r in plain}, {r['id']: r for r in enhanced}
    if len(before) != len(plain) or len(after) != len(enhanced) or before.keys() != after.keys():
        raise ValueError('Shared calibration requires unique matched source development predictions')
    if any(before[key]['label'] != after[key]['label'] for key in before):
        raise ValueError('Shared calibration labels changed within a development pair')
    valid = [[{k: v for k, v in r.items() if k != 'threshold'} for r in rows
              if r['status'] == 'ok' and r.get('probability_valid', True)] for rows in (plain, enhanced)]
    if not valid[0]:
        raise ValueError('Source plain development predictions are required for the shared cutoff')
    cut = threshold(valid[0])
    identity = grpo.signature([{k: r[k] for k in ('id', 'family', 'label', 'p', 'status')}
                               for r in sorted(valid[0], key=lambda r: (r['family'], r['id']))])
    return cut, {'selection': 'seed_shared_source_dev', 'target_adaptation': False,
                 'selection_condition': 'plain_seed_only', 'seed_prediction_signature': identity,
                 'shared_cutoff': cut, 'source_n': len(plain), 'selection_n': len(valid[0]),
                 'seed': classification(valid[0], cut),
                 'ega': classification(valid[1], cut) if valid[1] else None,
                 'excluded': {'seed': len(plain)-len(valid[0]), 'ega': len(enhanced)-len(valid[1])}}


def _validate_source_predictions(rows, records, name):
    expected = {(r['family'], r['id']): r for r in records}
    actual = {(r['family'], r['id']): r for r in rows}
    if len(actual) != len(rows) or actual.keys() != expected.keys():
        raise ValueError(f'Fitted {name} predictions do not cover the original source development cohort')
    if any(actual[key]['label'] != rec['label'] or actual[key]['participant'] != rec['participant']
           for key, rec in expected.items()):
        raise ValueError(f'Fitted {name} source participant labels changed')


def bundle_budget_policy(bundle, requested=None):
    """Keep the fitted source time unit identical during restore and reuse."""
    from .budget_units import budget_normalization_mode, budget_reference
    policy, fitting = bundle['policy'], bundle['fitting']
    mode = budget_normalization_mode(policy)
    if (budget_normalization_mode(fitting) != mode
            or (requested is not None and budget_normalization_mode(requested) != mode)):
        raise ValueError('Controller budget normalization differs from its fitted policy')
    if mode == 'per_seed_wall':
        return {}
    profile = policy.get('budget_profile')
    if (profile != fitting.get('budget_profile')
            or (requested is not None and requested.get('budget_profile') is not None
                and requested['budget_profile'] != profile)):
        raise ValueError('Controller source budget profile changed')
    budget_reference(dict(policy, **({'budget': requested['budget']}
                                     if requested is not None and 'budget' in requested else {})), 1.)
    return {'budget_normalization': mode, 'budget_profile': profile}


def bundle_target_policy(bundle, requested=None):
    """Restore the fitted TRAIN-only regression unit without fitting a new scale."""
    from ..controller import target_normalization_mode, validate_target_profile
    policy, fitting = bundle['policy'], bundle['fitting']
    mode = target_normalization_mode(policy)
    if (target_normalization_mode(fitting) != mode
            or (requested is not None and target_normalization_mode(requested) != mode)):
        raise ValueError('Controller target normalization differs from its fitted policy')
    if (mode != 'none') != ('target_scale' in bundle['controller']):
        raise ValueError('Controller target scale buffer differs from its fitted policy')
    if mode == 'none':
        return {}
    profile = validate_target_profile(policy.get('target_profile'))
    if (profile is None or profile['mode'] != mode or profile != fitting.get('target_profile')
            or (requested is not None and requested.get('target_profile') is not None
                and requested['target_profile'] != profile)):
        raise ValueError('Controller source target scale changed')
    return {'target_normalization': mode, 'target_profile': profile}


def _cached_policy_predictions(backbone, records, track, controller, policy, config, path, resume):
    expected = grpo.signature((backbone.spec, backbone.reader.fingerprint, _discriminator_fingerprint(backbone),
                               controller.state_dict(), policy, track, config, _record_signature(records)))
    cached = grpo.load_checkpoint(path, resume, 'ega-policy-calibration-v1', expected) if path else None
    before, after = (cached['seed'], cached['ega']) if cached else ([], [])
    done = {r['id'] for r in before}
    for rec in records:
        if rec['id'] in done:
            continue
        initial, final = acquire(backbone, rec, track, controller, policy, config.get('budget', 8),
                                 config.get('acquisition_limit', 8))
        before.append(row(rec, initial))
        after.append(row(rec, final))
        if path:
            grpo.save_checkpoint(path, {'schema': 'ega-policy-calibration-v1', 'signature': expected,
                                        'seed': before, 'ega': after})
        print(json.dumps({'stage': 'source_dev_calibration', 'record': rec['id'],
                           'done': len(before), 'total': len(records)}), flush=True)
    return before, after


def _participant_roster(records):
    return sorted({r['family']+':'+r['participant'] for r in records})


def _reader_source_refit(origin, frontend_origin, train, dev, reader_train, reader_dev):
    """Separate the verified old Reader roster from one declared source extension."""
    from .reader_sources import refit_declaration
    if (origin.get('schema') != 'ega-reader-source-refit-v1'
            or origin.get('original_sources') != ['daic']
            or origin.get('new_sources') != ['daic', 'mmpsy_depression']
            or not frontend_origin
            or origin.get('frontend_checkpoint_sha256') != frontend_origin.get('checkpoint_sha256')):
        raise ValueError('Reader source refitting requires its verified DAIC frontend declaration')
    original_train, original_dev = origin.get('original_reader_train'), origin.get('original_reader_dev')
    if (not isinstance(original_train, list) or not isinstance(original_dev, list)
            or grpo.signature((_record_signature(original_train), _record_signature(original_dev)))
            != grpo.signature((_record_signature(train), _record_signature(dev)))):
        raise ValueError('Reader source extension must retain the exact native DAIC source records')
    supplement = origin.get('supplement_provenance') or {}
    if (supplement.get('task') != 'mmpsy_depression'
            or supplement.get('partitions') != ['train', 'dev'] or supplement.get('modalities') != ['T']
            or not all(supplement.get(key) for key in ('manifest_sha256', 'partition_sha256', 'records_sha256'))
            or not origin.get('original_input_provenance')):
        raise ValueError('Reader supplement requires its frozen source TRAIN/DEV text provenance')
    for split, original, records in (('train', original_train, reader_train), ('dev', original_dev, reader_dev)):
        added = [r for r in records if r['family'] == 'mmpsy']
        if (not added or any(r['family'] != 'daic' or r['split'] != split for r in original)
                or any(r['split'] != split or any(v['modality'] != 'T' for v in r['views']) for r in added)
                or _record_signature(records) != _record_signature(original+added)
                or len({(r['family'], r['id']) for r in records}) != len(records)):
            raise ValueError('Expanded Reader records must preserve DAIC and add only declared MMPsy source text')
    if set(_participant_roster(reader_train)) & set(_participant_roster(reader_dev)):
        raise ValueError('Expanded Reader TRAIN/DEV participants overlap')
    return original_train, original_dev, refit_declaration(origin)


def validate_joint_reader_refit(original, requested):
    """Only the two audited decoder changes may refit an existing source Reader."""
    if (original.get('boundary_likelihood') != 'ordered_conditional_stop_v1'
            or original.get('speaker_normalization', 'literal') != 'literal'
            or original.get('relation_conditioning') != 'predicted_span_mean'):
        raise ValueError('Joint Reader refitting requires the verified ordered-stop span Reader')
    expected = dict(original, boundary_likelihood='global_legal_joint_v1',
                    speaker_normalization='ontology_casefold_v1')
    if requested != expected:
        raise ValueError('Joint Reader refitting may change only boundary likelihood and speaker normalization')


def validate_reader_optimizer_refit(original, requested):
    """Refit the verified global/casefold Reader with only the approved optimizer budget."""
    if (original.get('boundary_likelihood') != 'global_legal_joint_v1'
            or original.get('speaker_normalization') != 'ontology_casefold_v1'
            or original.get('training_mode', 'fullbatch') != 'fullbatch'):
        raise ValueError('Minibatch refitting requires the verified fullbatch global/casefold Reader')
    expected = dict(original, training_mode='participant_minibatch_v1', participant_batch_size=16)
    if requested != expected:
        raise ValueError('Minibatch Reader refitting may change only its approved training mode and batch size')


def frozen_component_identity(bundle, spec, train, dev, reader_train, reader_dev, track, config, seed,
                              reader_refit=False):
    """Validate only the immutable frontend; the old policy and targets are unused."""
    if bundle.get('schema') != 'ega-v2' or not bundle.get('reader') or not bundle.get('discriminator'):
        raise ValueError('Frontend reuse requires a complete Reader and Screening discriminator checkpoint')
    reader, discriminator = bundle['reader'], bundle['discriminator']
    if reader_refit and config['reader'].get('training_mode') == 'participant_minibatch_v1':
        validate_reader_optimizer_refit(reader['config'], config['reader'])
    elif reader_refit and (config['reader'].get('boundary_likelihood') == 'global_legal_joint_v1'
                        or config['reader'].get('speaker_normalization') == 'ontology_casefold_v1'):
        validate_joint_reader_refit(reader['config'], config['reader'])
    if reader.get('backbone_spec') != spec or discriminator.get('backbone_spec') != spec:
        raise ValueError('Frontend checkpoint requires the same foundation model/Screening configuration')
    from ..discriminator import _validate
    if ((not reader_refit and reader['config'] != config['reader'])
            or discriminator['config'] != _validate(config['discriminator'])):
        raise ValueError('Frontend training configuration changed; retrain Reader/discriminator')
    if reader.get('seed') != seed or discriminator.get('seed') != seed:
        raise ValueError('Frontend checkpoint requires its original training seed')
    if discriminator.get('reader_fingerprint') != reader.get('fingerprint'):
        provenance = bundle.get('reader_refit_provenance')
        if (not provenance or provenance.get('fingerprint') != digest(
                {k: v for k, v in provenance.items() if k != 'fingerprint'})
                or provenance.get('original_reader') != discriminator.get('reader_fingerprint')
                or provenance.get('reader') != reader.get('fingerprint')
                or provenance.get('discriminator') != discriminator.get('fingerprint')
                or provenance.get('source_records') != grpo.signature(
                    (_record_signature(reader_train), _record_signature(reader_dev)))):
            raise ValueError('Changed Reader requires its declared source-only refit provenance')
        if provenance.get('source_declaration'):
            declaration = provenance['source_declaration']
            native_reader = ([r for r in reader_train if r['family'] == 'daic'],
                             [r for r in reader_dev if r['family'] == 'daic'])
            if (declaration.get('schema') != 'ega-reader-source-refit-v1'
                    or declaration.get('new_sources') != ['daic', 'mmpsy_depression']
                    or provenance.get('original_source_records') != grpo.signature(
                        tuple(_record_signature(records) for records in native_reader))
                    or provenance.get('reader_participants') != {
                        'train': _participant_roster(reader_train), 'dev': _participant_roster(reader_dev)}):
                raise ValueError('Reader source extension does not match its original DAIC and new source rosters')
    for split, records, reader_records in (('train', train, reader_train), ('dev', dev, reader_dev)):
        reader_roster = sorted(':'.join(p) for p in reader[split+'_participants'])
        if (bundle.get(split+'_participants') != _participant_roster(records)
                or reader_roster != _participant_roster(reader_records)):
            raise ValueError('Frontend checkpoint source participant partitions changed')
    adapter = bundle.get('adapter')
    if (track == 'F') != bool(adapter):
        raise ValueError('Frontend feature adapter does not match this input track')
    return {'reader': reader['fingerprint'], 'discriminator': discriminator['fingerprint'],
            'adapter': grpo.signature((adapter['stats'], adapter['state_dict'])) if adapter else None}


def teacher_generation_config(config, include_augmentation=True):
    """Only fields that produce frozen targets, independent of Controller loss."""
    defaults = {'max_teacher_pairs': None, 'max_teacher_states': None, 'budget': 8,
                'reader_gate': None, 'state_representation': 'hash',
                'seed_prediction_anchor': 'legacy_reader', 'state_evidence': 'gated'}
    keys = ('seed', 'prefix_lengths', 'actions_per_state', *defaults)
    from ..controller import (gain_loss_settings, teacher_augmentation_settings, teacher_sampling_settings,
                      controller_state_evidence_mode)
    from .teacher_rebuild import teacher_rebuild_settings
    from .score_calibration import calibration_settings
    from .controller_candidates import candidate_conditioning_mode
    from .controller_rollouts import teacher_rollout_settings
    value = {key: config.get(key, defaults.get(key)) for key in keys} | {
        'gain_loss': gain_loss_settings(config), 'teacher_sampling': teacher_sampling_settings(config),
        'teacher_input_rebuild': teacher_rebuild_settings(config),
        'score_calibration': calibration_settings(config)}
    value['state_evidence'] = controller_state_evidence_mode(config)
    value['candidate_conditioning'] = candidate_conditioning_mode(config)
    value['teacher_rollout'] = teacher_rollout_settings(config)
    if include_augmentation:
        value['teacher_augmentation'] = teacher_augmentation_settings(config)
    return value


def load_teacher_cache(checkpoint, bundle, train, dev, track, config, original_controller, origin, component_ids):
    """Read completed per-source-record targets, never run the frozen LLM."""
    directory = Path(checkpoint).parent
    protocol_bytes = (directory/'fit_protocol.json').read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol.get('schema') != 'ega-fit-v1' or protocol.get('signature') != bundle.get('fit_signature'):
        raise ValueError('Teacher origin final checkpoint does not match its fitting protocol')
    original_controller = protocol.get('teacher_config', original_controller)
    requested_loss = config['controller'].get('gain_loss', {})
    recorded_loss = original_controller.get('gain_loss', {})
    if (requested_loss.get('mode') == recorded_loss.get('mode') == 'source_cut_centered_squared'
            and 'source_cut' not in requested_loss and 'source_cut' in recorded_loss):
        # Its source/frontend identity was checked before loading these targets.
        config['controller'] = dict(config['controller'], gain_loss=dict(recorded_loss))
    gate = bundle.get('policy', {}).get('reader_gate')
    if gate is not None and gate.get('fingerprint') != digest({k: v for k, v in gate.items() if k != 'fingerprint'}):
        raise ValueError('Teacher origin Reader gate fingerprint mismatch')
    original_controller = dict(original_controller, reader_gate=gate) if gate is not None else dict(original_controller)
    original_calibration = original_controller.get('score_calibration', {'mode': 'disabled'})
    requested_calibration = config['controller'].get('score_calibration', {'mode': 'disabled'})
    if (original_calibration.get('mode', 'disabled') != 'disabled'
            and requested_calibration == {'mode': original_calibration['mode']}):
        config['controller'] = dict(config['controller'], score_calibration=original_calibration)
    current_controller = dict(config['controller'], reader_gate=gate) if gate is not None else config['controller']
    generation = teacher_generation_config(original_controller)
    requested_generation = teacher_generation_config(current_controller)
    from .teacher_rebuild import sanctioned_rebuild
    migration = sanctioned_rebuild(generation, requested_generation)
    score_migration = (generation['score_calibration']['mode'] == 'disabled'
                       and requested_generation['score_calibration']['mode'] == 'source_audio_cut_transport')
    evidence_migration = (generation['state_evidence'] == 'gated'
                          and requested_generation['state_evidence'] == 'observed_reader')
    candidate_migration = (generation['candidate_conditioning'] == 'none'
                           and requested_generation['candidate_conditioning'] == 'observed_event')
    if candidate_migration and (migration or score_migration or evidence_migration
            or generation['state_evidence'] != 'observed_reader'
            or {k: v for k, v in generation.items() if k != 'candidate_conditioning'}
               != {k: v for k, v in requested_generation.items() if k != 'candidate_conditioning'}):
        raise ValueError('Candidate mask migration requires the unchanged completed observed-Reader teachers')
    if evidence_migration and (migration or gate is None or gate['accept']
            or generation['state_representation'] != 'frozen_discriminator_context'
            or generation['seed_prediction_anchor'] != 'legacy_reader'
            or current_controller.get('grpo', {}).get('enabled', True)):
        raise ValueError('Observed Reader migration requires the unchanged Reader-rejected native-context source teachers')
    if score_migration:
        allowed = {'score_calibration', 'teacher_augmentation'} | ({'state_evidence'} if evidence_migration else set())
        if (migration or {k: v for k, v in generation.items() if k not in allowed}
                != {k: v for k, v in requested_generation.items() if k not in allowed}
                or generation['state_representation'] != 'frozen_discriminator_context'
                or generation['seed_prediction_anchor'] != 'legacy_reader' or gate is None or gate['accept']
                or requested_generation['teacher_augmentation']['mode'] != 'disabled'):
            raise ValueError('Audio transport requires the unchanged native-context, Reader-rejected source teachers')
    ignored = ({'teacher_augmentation'} | ({'state_evidence'} if evidence_migration else set())
               | ({'candidate_conditioning'} if candidate_migration else set()))
    if not migration and not score_migration and ({k: v for k, v in generation.items() if k not in ignored}
                          != {k: v for k, v in requested_generation.items() if k not in ignored}):
        raise ValueError('Teacher generation settings changed; regenerate frozen targets')
    if (migration or score_migration or evidence_migration or candidate_migration) and any(current_controller.get(k, default) != original_controller.get(k, default)
                         for k, default in (('policy_lambda', .01), ('policy_gamma', .1))):
        raise ValueError('Input rebuild cannot change the original utility weights')
    if config.get('calibration', {}).get('mode') == 'joint_source_dev' and gate is None:
        raise ValueError('Teacher checkpoint is missing its source-development Reader gate')
    examples, timings, files = [[], []], [], {}
    for split_index, (split, records) in enumerate((('train', train), ('dev', dev))):
        for rec in records:
            key = digest((rec['family'], rec['id']))[:20]
            relative = f'teacher/{split}/{key}.pt'
            content = (directory/relative).read_bytes()
            cached = torch.load(io.BytesIO(content), map_location='cpu', weights_only=False)
            expected = grpo.signature((component_ids['reader'], component_ids['discriminator'],
                                       track, original_controller, _record_signature([rec])))
            if cached.get('schema') != 'ega-teacher-record-v1' or cached.get('signature') != expected:
                raise ValueError(f'Teacher record does not match its source/frontend/config: {relative}')
            examples[split_index].extend(cached['pairs'])
            if split == 'train':
                timings.extend(cached['timings'])
            files[relative] = hashlib.sha256(content).hexdigest()
    if len(examples[0]) != bundle.get('teacher_pair_count'):
        raise ValueError('Teacher checkpoint does not cover its declared training targets')
    for example in examples[0]+examples[1]:
        if (example.get('reader_fingerprint') != component_ids['reader']
                or example.get('discriminator_fingerprint') != component_ids['discriminator']
                or example.get('reader_gate_signature') != reader_gate_signature(gate)
                or example.get('cost_basis') != COST_BASIS):
            raise ValueError('Teacher target frontend/gate/cost identity changed')
        for field, default in (('state_representation', 'hash'), ('seed_prediction_anchor', 'legacy_reader'),
                               ('state_evidence', 'gated'), ('candidate_conditioning', 'none')):
            if example.get(field, default) != generation[field]:
                raise ValueError('Teacher target Controller input mode changed')
        if generation['state_representation'] == 'frozen_token_embeddings' and (
                example.get('state_representation_version') != STATE_REPRESENTATION_VERSION):
            raise ValueError('Teacher target state representation version changed')
    if not timings or any(len(t) != 4 or t[-1] != COST_BASIS for t in timings):
        raise ValueError('Teacher timing cache does not use the current cost basis')
    provenance = dict(origin, components=component_ids, files=files, generation_config=generation,
                      reader_gate=gate, fit_signature=bundle['fit_signature'],
                      fit_protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
                      source_records=grpo.signature((_record_signature(train), _record_signature(dev))))
    if migration:
        provenance.update(input_rebuild_required=True, rebuild_generation_config=requested_generation)
    if score_migration:
        provenance.update(score_calibration_required=True)
    if evidence_migration:
        provenance.update(state_evidence_required=True)
    if candidate_migration:
        provenance.update(candidate_conditioning_required=True)
    if bundle.get('dev_seed_predictions'):
        plain = bundle['dev_seed_predictions']
        _validate_source_predictions(plain, dev, 'teacher-origin source plain predictions')
        if any(r.get('status') != 'ok' or not r.get('probability_valid', False)
               or not isinstance(r.get('p'), (int, float)) or not 0. <= r['p'] <= 1.
               or r.get('evidence') or r.get('evidence_tokens', 0) != 0 or r.get('reader_used', False)
               for r in plain):
            raise ValueError('Teacher-origin source plain predictions must contain valid unaugmented probabilities')
        provenance['source_plain_predictions'] = plain
        provenance['source_plain_signature'] = grpo.signature(plain)
    return (examples[0], examples[1], timings), provenance


def _rebuild_cached_targets(backbone, train, dev, track, config, pair_cache, origin, directory, resume):
    """Persist derived records/state contexts; the verified base files remain read-only."""
    from .teacher_rebuild import rebuild_record_targets, teacher_rebuild_settings
    if not origin.get('input_rebuild_required') or directory is None:
        raise ValueError('Input rebuilding requires verified base targets and a saved run directory')
    profile = teacher_rebuild_settings(config)
    merged, files, counts = [], {}, {}
    base_directory = Path(origin['checkpoint']).parent
    for split, records, base_pairs in (('train', train, pair_cache[0]), ('dev', dev, pair_cache[1] or [])):
        derived, statistics = [], {}
        for index, rec in enumerate(records):
            key = digest((rec['family'], rec['id']))[:20]
            relative = f'teacher/{split}/{key}.pt'
            originals = [e for e in base_pairs if e['participant'] == rec['family']+':'+rec['participant']
                         and e['state_id'][0] == rec['id']]
            expected = grpo.signature((origin['checkpoint_sha256'], origin['files'][relative],
                                       origin['components'], _record_signature([rec]), track,
                                       teacher_generation_config(config), originals))
            path = directory/'teacher_rebuild'/split/(key+'.pt')
            cached = grpo.load_checkpoint(path, resume, 'ega-teacher-rebuild-record-v1', expected)
            hit = cached is not None
            if cached is None:
                pairs, stats = rebuild_record_targets(backbone, rec, track, config, originals,
                    directory/'teacher_rebuild'/split/key, resume, expected)
                original = torch.load(base_directory/relative, map_location='cpu', weights_only=False)
                cached = {'schema': 'ega-teacher-rebuild-record-v1', 'signature': expected,
                          'pairs': pairs, 'timings': original['timings'], 'stats': stats,
                          'excluded': original.get('excluded')}
                grpo.save_checkpoint(path, cached)
            derived.extend(cached['pairs'])
            statistics[key] = cached['stats']
            files[str(path.relative_to(directory)).replace('\\', '/')] = hashlib.sha256(path.read_bytes()).hexdigest()
            target_signature = grpo.signature((backbone.reader.fingerprint, _discriminator_fingerprint(backbone),
                                               track, config, _record_signature([rec])))
            grpo.save_checkpoint(directory/relative, {'schema': 'ega-teacher-record-v1',
                'signature': target_signature, 'pairs': cached['pairs'], 'timings': cached['timings'],
                'excluded': cached['excluded']})
            stats = cached['stats']
            print('teacher_rebuild_progress: '+json.dumps({'split': split, 'record': rec['id'],
                'done': index+1, 'total': len(records), 'pairs': len(cached['pairs']),
                'initial_states': stats['initial_states'], 'context_states': stats['context_states'],
                'cache_reused_states': stats['context_states'] if hit else stats['cache_hits'],
                'canonical_before_scores': 0 if hit else stats['canonical_before_scores'],
                'initial_variant_before_scores': 0 if hit else stats['initial_variant_before_scores'],
                'after_scores': 0, 'initial_extra_delta_seconds':
                    stats.get('budget_migration', {}).get('initial_extra_delta_seconds', 0.)}), flush=True)
        merged.append(derived)
        counts[split] = {'base_pairs': len(base_pairs), 'derived_pairs': len(derived), 'records': statistics}
    provenance = {'config': profile, 'base_checkpoint_sha256': origin['checkpoint_sha256'],
                  'base_fit_signature': origin['fit_signature'], 'base_files': origin['files'],
                  'files': files, 'counts': counts, 'after_scores': 0,
                  'timing_cost_profile': 'unchanged_base_timings',
                  'runtime_budget_migration': 'seed_reader_extra_delta_v1'}
    return (merged[0], merged[1], pair_cache[2]), provenance


def _calibrate_cached_targets(backbone, train, dev, track, config, pair_cache, origin, directory, resume,
                              raw_directory=None):
    """Transport cached raw probabilities; native inputs/timings never require rescoring."""
    from .score_calibration import calibrated_record_targets, validate_profile
    profile = validate_profile(config['score_calibration'])
    fresh = raw_directory is not None
    if directory is None or (fresh and origin is not None) or (not fresh and not origin.get('score_calibration_required')):
        raise ValueError('Audio transport requires verified raw teachers and a saved output directory')
    base_directory = Path(raw_directory) if fresh else Path(origin['checkpoint']).parent
    merged, files, raw_files = [], {}, {}
    for split, records, old_pairs in (('train', train, pair_cache[0]), ('dev', dev, pair_cache[1] or [])):
        values = []
        for index, rec in enumerate(records):
            key = digest((rec['family'], rec['id']))[:20]
            relative = f'teacher/{split}/{key}.pt'
            original = [e for e in old_pairs if e['participant'] == rec['family']+':'+rec['participant']
                        and e['state_id'][0] == rec['id']]
            base_path = base_directory/split/(key+'.pt') if fresh else base_directory/relative
            if fresh:
                raw_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
                raw_files[str(base_path.relative_to(directory)).replace('\\', '/')] = raw_sha
                identity = grpo.signature(('fresh-source-raw-teacher-v1', raw_sha,
                    backbone.reader.fingerprint, _discriminator_fingerprint(backbone),
                    _record_signature([rec]), track, teacher_generation_config(config), original))
            else:
                identity = grpo.signature((origin['checkpoint_sha256'], origin['files'][relative],
                    origin['components'], _record_signature([rec]), track, teacher_generation_config(config), original))
            path = directory/'teacher_score_calibration'/split/(key+'.pt')
            cached = grpo.load_checkpoint(path, resume, 'ega-score-calibrated-teacher-v1', identity)
            if cached is None:
                pairs = calibrated_record_targets(rec, original, profile, config['gain_loss'])
                base = torch.load(base_path, map_location='cpu', weights_only=False)
                cached = {'schema': 'ega-score-calibrated-teacher-v1', 'signature': identity,
                          'pairs': pairs, 'timings': base['timings'], 'excluded': base.get('excluded')}
                grpo.save_checkpoint(path, cached)
            values.extend(cached['pairs'])
            files[str(path.relative_to(directory)).replace('\\', '/')] = hashlib.sha256(path.read_bytes()).hexdigest()
            target_signature = grpo.signature((backbone.reader.fingerprint, _discriminator_fingerprint(backbone),
                                               track, config, _record_signature([rec])))
            grpo.save_checkpoint(directory/relative, {'schema': 'ega-teacher-record-v1',
                'signature': target_signature, 'pairs': cached['pairs'], 'timings': cached['timings'],
                'excluded': cached['excluded']})
            print('teacher_score_calibration_progress: '+json.dumps({'split': split, 'done': index+1,
                'total': len(records), 'pairs': len(cached['pairs']), 'new_model_scores': 0}), flush=True)
        merged.append(values)
    provenance = ({'mode': 'fresh_source_raw_teacher_transport', 'profile': profile, 'raw_files': raw_files,
                   'source_records_signature': grpo.signature((_record_signature(train), _record_signature(dev))),
                   'reader_fingerprint': backbone.reader.fingerprint,
                   'discriminator_fingerprint': _discriminator_fingerprint(backbone),
                   'derived_files': files, 'new_model_scores': 0} if fresh else
                  {'profile': profile, 'base_checkpoint_sha256': origin['checkpoint_sha256'],
                   'base_fit_signature': origin['fit_signature'], 'base_files': origin['files'],
                   'derived_files': files, 'new_model_scores': 0})
    return (merged[0], merged[1], pair_cache[2]), provenance


def _augment_cached_targets(backbone, train, dev, track, config, pair_cache, origin, directory, resume):
    """Keep verified base targets and persist only the missing source actions."""
    from ..controller import augment_teacher_pairs, teacher_augmentation_settings
    augmentation = teacher_augmentation_settings(config)
    if augmentation['mode'] == 'disabled':
        return pair_cache, None
    if not origin or not origin.get('checkpoint_sha256') or directory is None:
        raise ValueError('Teacher augmentation requires a verified base teacher checkpoint and a saved run directory')
    base_directory = Path(origin['checkpoint']).parent
    merged, files, counts = [], {}, {}
    new_train_timings = []
    for split, records, base_pairs in (('train', train, pair_cache[0]), ('dev', dev, pair_cache[1] or [])):
        augmentation_directory = directory/'teacher_augmentation'/split
        additions, new_timings, record_stats = [], [], {}
        # Write complete per-record targets for subsequent verified reuse. This
        # copies the original pairs; it never evaluates an original observation.
        for index, rec in enumerate(records):
            key = digest((rec['family'], rec['id']))[:20]
            augmentation_path = augmentation_directory/(key+'.pt')
            original_pairs = [e for e in base_pairs if e['participant'] == rec['family']+':'+rec['participant']
                              and e['state_id'][0] == rec['id']]
            base_relative = f'teacher/{split}/{key}.pt'
            expected = grpo.signature((backbone.reader.fingerprint, _discriminator_fingerprint(backbone),
                                       track, teacher_generation_config(config), _record_signature([rec]),
                                       origin['checkpoint_sha256'], origin['files'][base_relative],
                                       original_pairs))
            cached = grpo.load_checkpoint(augmentation_path, resume, 'ega-teacher-augmentation-record-v1', expected)
            if cached is None:
                added, measured = augment_teacher_pairs(backbone, [rec], track, config, original_pairs)
                _mark_discriminator_targets(added, backbone)
                cached = {'schema': 'ega-teacher-augmentation-record-v1', 'signature': expected,
                          'pairs': added, 'timings': measured,
                          'stats': getattr(backbone, 'teacher_augmentation_stats', None)}
                grpo.save_checkpoint(augmentation_path, cached)
            additions.extend(cached['pairs'])
            new_timings.extend(cached['timings'])
            record_stats[key] = cached.get('stats')
            content = augmentation_path.read_bytes()
            relative = str(augmentation_path.relative_to(directory)).replace('\\', '/')
            files[relative] = hashlib.sha256(content).hexdigest()
            original = torch.load(base_directory/base_relative, map_location='cpu', weights_only=False)
            expected = grpo.signature((backbone.reader.fingerprint, _discriminator_fingerprint(backbone),
                                       track, config, _record_signature([rec])))
            grpo.save_checkpoint(directory/base_relative, {
                'schema': 'ega-teacher-record-v1', 'signature': expected, 'pairs': original_pairs+cached['pairs'],
                'timings': original['timings']+cached['timings'], 'excluded': original.get('excluded')})
            print('teacher_augmentation_progress: '+json.dumps({
                'record': rec['id'], 'split': split, 'done': index+1, 'total': len(records),
                'base_pairs': len(original_pairs), 'added_pairs': len(cached['pairs'])}), flush=True)
        merged.append(base_pairs+additions)
        counts[split] = {'base_pairs': len(base_pairs), 'added_pairs': len(additions),
                         'merged_pairs': len(base_pairs)+len(additions), 'records': record_stats}
        if split == 'train':
            new_train_timings = new_timings
    provenance = {'config': augmentation, 'base_checkpoint_sha256': origin['checkpoint_sha256'],
                  'base_fit_signature': origin['fit_signature'], 'base_files': origin['files'],
                  'files': files, 'counts': counts}
    return (merged[0], merged[1], pair_cache[2]+new_train_timings), provenance


def _observed_reader_cached_targets(backbone, train, dev, track, config, pair_cache, origin, directory, resume):
    """Re-encode source states with frozen Reader records; leave all scores/targets intact."""
    from .controller_evidence import (BUDGET_BASIS, add_reader_timings, derive_record_pairs,
        encode_record_states, reader_cost_model)
    if not origin.get('state_evidence_required') or directory is None:
        raise ValueError('Observed Reader states require verified source caches and an output directory')
    base_directory = Path(origin['checkpoint']).parent
    work, measured = {}, []
    for split, records, base_pairs in (('train', train, pair_cache[0]), ('dev', dev, pair_cache[1] or [])):
        work[split] = []
        for index, rec in enumerate(records):
            key = digest((rec['family'], rec['id']))[:20]
            relative = f'teacher/{split}/{key}.pt'
            original = [e for e in base_pairs if e['participant'] == rec['family']+':'+rec['participant']
                        and e['state_id'][0] == rec['id']]
            identity = grpo.signature((origin['checkpoint_sha256'], origin['files'][relative],
                origin['components'], _record_signature([rec]), track, teacher_generation_config(config), original))
            states, reader_timings, stats = encode_record_states(backbone, rec, track, config, original,
                directory/'teacher_state_evidence'/split/key, resume, identity)
            base = torch.load(base_directory/relative, map_location='cpu', weights_only=False)
            work[split].append((rec, key, relative, original, identity, states, stats, base))
            if split == 'train':
                measured.extend(reader_timings)
            print('teacher_state_evidence_progress: '+json.dumps({'split': split, 'done': index+1,
                'total': len(records), 'states': stats['states'], 'cache_hits': stats['cache_hits'],
                'new_llm_scores': 0}), flush=True)
    reader_cost = reader_cost_model(measured)
    merged, new_timings, files, counts = [], [], {}, {}
    for split in ('train', 'dev'):
        pairs, records = [], {}
        for rec, key, relative, original, identity, states, stats, base in work[split]:
            path = directory/'teacher_state_evidence'/split/(key+'.pt')
            signature = grpo.signature((identity, reader_cost, states))
            cached = grpo.load_checkpoint(path, resume, 'ega-controller-evidence-record-v1', signature)
            if cached is None:
                cached = {'schema': 'ega-controller-evidence-record-v1', 'signature': signature,
                    'pairs': derive_record_pairs(rec, track, config, original, states, reader_cost),
                    'timings': add_reader_timings(base['timings'], reader_cost),
                    'original_timings': base['timings'], 'reader_timings':
                        [v['cost_features'][1:]+[v['seconds'], COST_BASIS] for v in states.values()],
                    'excluded': base.get('excluded')}
                grpo.save_checkpoint(path, cached)
            pairs.extend(cached['pairs'])
            if split == 'train':
                new_timings.extend(cached['timings'])
            files[str(path.relative_to(directory)).replace('\\', '/')] = hashlib.sha256(path.read_bytes()).hexdigest()
            records[key] = {'states': stats['states'], 'pairs': len(cached['pairs'])}
            target_signature = grpo.signature((backbone.reader.fingerprint, _discriminator_fingerprint(backbone),
                                               track, config, _record_signature([rec])))
            grpo.save_checkpoint(directory/relative, {'schema': 'ega-teacher-record-v1',
                'signature': target_signature, 'pairs': cached['pairs'], 'timings': cached['timings'],
                'original_timings': cached['original_timings'], 'reader_timings': cached['reader_timings'],
                'excluded': cached['excluded']})
        merged.append(pairs)
        counts[split] = records
    provenance = {'mode': 'observed_reader', 'base_checkpoint_sha256': origin['checkpoint_sha256'],
        'base_files': origin['files'], 'derived_files': files, 'reader_cost': reader_cost,
        'reader_cost_fitted_on': 'source_training_observed_states', 'budget_basis': BUDGET_BASIS,
        'counts': counts, 'new_llm_scores': 0, 'original_raw_timings_preserved': True}
    return (merged[0], merged[1], new_timings), provenance


def _candidate_conditioned_cached_targets(backbone, train, dev, track, config, pair_cache, origin, directory, resume):
    """Derive only candidate masks from verified completed source inputs."""
    from .controller_evidence import derive_candidate_pairs
    if not origin.get('candidate_conditioning_required') or directory is None:
        raise ValueError('Candidate masks require verified completed source teachers and an output directory')
    base_directory = Path(origin['checkpoint']).parent
    merged, files, counts = [], {}, {}
    for split, records, base_pairs in (('train', train, pair_cache[0]), ('dev', dev, pair_cache[1] or [])):
        pairs, count = [], {'records': 0, 'pairs': 0, 'event_matched_pairs': 0, 'cache_hits': 0}
        for index, rec in enumerate(records):
            key = digest((rec['family'], rec['id']))[:20]
            relative = f'teacher/{split}/{key}.pt'
            original = [e for e in base_pairs if e['participant'] == rec['family']+':'+rec['participant']
                        and e['state_id'][0] == rec['id']]
            identity = grpo.signature((origin['checkpoint_sha256'], origin['files'][relative],
                origin['components'], _record_signature([rec]), track, teacher_generation_config(config), original))
            path = directory/'teacher_candidate_conditioning'/split/(key+'.pt')
            cached = grpo.load_checkpoint(path, resume, 'ega-controller-candidate-record-v1', identity)
            if cached is None:
                cached = {'schema': 'ega-controller-candidate-record-v1', 'signature': identity,
                          'pairs': derive_candidate_pairs(rec, config, original)}
                grpo.save_checkpoint(path, cached)
            else:
                count['cache_hits'] += 1
            pairs.extend(cached['pairs'])
            count['records'] += 1
            count['pairs'] += len(cached['pairs'])
            count['event_matched_pairs'] += sum(bool(e['candidate_mask'].any()) for e in cached['pairs'])
            files[str(path.relative_to(directory)).replace('\\', '/')] = hashlib.sha256(path.read_bytes()).hexdigest()
            # Preserve the completed record's timings, exclusions and recovery metadata.
            base = torch.load(base_directory/relative, map_location='cpu', weights_only=False)
            target_signature = grpo.signature((backbone.reader.fingerprint, _discriminator_fingerprint(backbone),
                                               track, config, _record_signature([rec])))
            grpo.save_checkpoint(directory/relative, dict(base, signature=target_signature, pairs=cached['pairs']))
            print('teacher_candidate_conditioning_progress: '+json.dumps({'split': split, 'done': index+1,
                'total': len(records), 'pairs': count['pairs'], 'cache_hits': count['cache_hits'],
                'new_llm_scores': 0, 'new_reader_reads': 0}), flush=True)
        merged.append(pairs)
        counts[split] = {k: v for k, v in count.items() if k != 'cache_hits'}
    provenance = {'mode': 'observed_event', 'base_checkpoint_sha256': origin['checkpoint_sha256'],
        'base_files': origin['files'], 'derived_files': files, 'counts': counts,
        'new_llm_scores': 0, 'new_reader_reads': 0,
        'preserved': ['x', 'state', 'raw_probabilities', 'targets', 'timings', 'reader_records', 'budget']}
    return (merged[0], merged[1], pair_cache[2]), provenance


def fit_bundle(backbone, train, dev, track, config, variant='full', seed=27, selected=None, pair_cache=None,
               reader_train=None, reader_dev=None, checkpoint_dir=None, resume=False,
               frozen_components=None, frontend_origin=None, teacher_origin=None,
               trained_bundle=None, fitted_origin=None, reader_refit_origin=None, plain_teacher_origin=None):
    if not config.get('reader', {}).get('enabled', False):
        raise ValueError('Full EGA training requires the supervised evidence reader')
    reader_refit = config.get('reader_refit', False)
    if type(reader_refit) is not bool:
        raise ValueError('reader_refit must be an explicit boolean')
    if reader_refit and (frozen_components is None or pair_cache is not None
                         or teacher_origin is not None or trained_bundle is not None or selected is not None):
        raise ValueError('Reader refitting requires a frozen frontend and fresh source targets')
    from .score_calibration import calibration_settings, fit_source_audio, validate_profile
    from .controller_candidates import candidate_conditioning_mode, action_filter_mode
    from .reader_state import reader_state_mode
    from .controller_states import state_filter_mode, mark_record_states
    evidence_encoding = reader_state_mode(config['controller'])
    if plain_teacher_origin is not None and (
            (not reader_refit and evidence_encoding != 'structured_soft')
            or frozen_components is None or pair_cache is not None or teacher_origin is not None
            or trained_bundle is not None or selected is not None):
        raise ValueError('Plain Teacher derivation requires Reader refitting or frozen soft-state re-encoding')
    state_filter = state_filter_mode(config['controller'])
    candidate_mode = candidate_conditioning_mode(config['controller'])
    action_filter = action_filter_mode(config['controller'])
    if state_filter == 'seed_aligned_audio' and action_filter != 'aligned_audio':
        raise ValueError('Seed-aligned source states require the aligned-audio runtime policy')
    score_request = calibration_settings(config['controller'])
    from .teacher_rebuild import teacher_rebuild_settings
    rebuild_profile = teacher_rebuild_settings(config['controller'])
    if rebuild_profile['mode'] != 'disabled':
        config = dict(config, controller=dict(config['controller'], teacher_input_rebuild=rebuild_profile))
    discriminator_enabled = config.get('discriminator', {}).get('enabled', False)
    if discriminator_enabled and selected is not None and not selected.get('discriminator'):
        raise ValueError('Outer refitting requires the inner-selected discriminator epoch count')
    started = time.perf_counter()
    directory = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if plain_teacher_origin is not None and directory is None:
        raise ValueError('Plain Teacher derivation requires a saved source run directory')
    reader_train = train if reader_train is None else reader_train
    reader_dev = dev if reader_dev is None else reader_dev
    original_reader_train, original_reader_dev = reader_train, reader_dev
    reader_source_declaration = None
    if reader_refit_origin is not None:
        if not reader_refit:
            raise ValueError('A new Reader source declaration requires explicit Reader refitting')
        original_reader_train, original_reader_dev, reader_source_declaration = _reader_source_refit(
            reader_refit_origin, frontend_origin, train, dev, reader_train, reader_dev)
    elif reader_refit and frozen_components is not None:
        reader_source_declaration = (frozen_components.get('reader_refit_provenance') or {}).get('source_declaration')
    component_ids = None
    source_signature = grpo.signature((_record_signature(train), _record_signature(dev)))
    if trained_bundle is not None:
        from ..controller import controller_state_evidence_mode
        from .controller_ranking import ranking_settings
        bundle_budget_policy(trained_bundle, config['controller'])
        bundle_target_policy(trained_bundle, config['controller'])
        if evidence_encoding != reader_state_mode(trained_bundle['policy']):
            raise ValueError('Fitted reuse cannot change the trained Reader-state encoding')
        if ranking_settings(config['controller']) != trained_bundle['fitting'].get(
                'ranking_loss', ranking_settings({})):
            raise ValueError('Fitted reuse cannot change the trained ranking objective')
        if controller_state_evidence_mode(config['controller']) != controller_state_evidence_mode(trained_bundle['policy']):
            raise ValueError('Fitted reuse cannot change the Controller observed evidence mode')
        if candidate_mode != candidate_conditioning_mode(trained_bundle['policy']):
            raise ValueError('Fitted reuse cannot change Controller candidate conditioning')
        if action_filter != action_filter_mode(trained_bundle['policy']):
            raise ValueError('Fitted reuse cannot change Controller action filtering')
        if state_filter != state_filter_mode(trained_bundle['policy']):
            raise ValueError('Fitted reuse cannot change Controller state filtering')
        fitted_calibration = trained_bundle.get('policy', {}).get('score_calibration', {'mode': 'disabled'})
        if (score_request != fitted_calibration and score_request != {'mode': fitted_calibration['mode']}):
            raise ValueError('Fitted reuse cannot change the trained score calibration')
        if (not fitted_origin or not fitted_origin.get('checkpoint_sha256')
                or fitted_origin.get('fit_signature') != trained_bundle.get('fit_signature')):
            raise ValueError('Fitted reuse requires its hashed checkpoint and original fitting protocol')
        if any(x is not None for x in (selected, pair_cache, frozen_components, teacher_origin)):
            raise ValueError('Fitted calibration reuse cannot change or refit a trained component')
        if config.get('calibration', {}).get('threshold_objective') != 'seed_shared_source_dev':
            raise ValueError('Fitted calibration reuse requires the source-plain shared threshold objective')
        requested_loss = config['controller'].get('gain_loss', {})
        recorded_loss = trained_bundle.get('gain_loss', {'mode': 'raw_brier'})
        if (requested_loss.get('mode', 'raw_brier') != recorded_loss['mode']
                or ('source_cut' in requested_loss and requested_loss['source_cut'] != recorded_loss.get('source_cut'))):
            raise ValueError('Fitted calibration reuse cannot change the trained Controller target loss')
        from ..controller import gain_loss_settings
        gate_loss = config.get('calibration', {}).get('reader_gate_loss', 'raw_brier')
        expected_gate_loss = recorded_loss if gate_loss == 'controller_gain_loss' else {'mode': 'raw_brier'}
        if (gate_loss not in ('raw_brier', 'controller_gain_loss')
                or gain_loss_settings({'gain_loss': (trained_bundle['policy'].get('reader_gate') or {}).get(
                    'gain_loss', {})}) != expected_gate_loss):
            raise ValueError('Fitted reuse cannot change the trained Reader gate loss; regenerate paired targets')
        component_ids = frozen_component_identity(trained_bundle, backbone.spec, train, dev,
                                                  reader_train, reader_dev, track, config, seed)
        recorded_source = (trained_bundle.get('source_records_signature')
                           or (trained_bundle.get('teacher_origin') or {}).get('source_records'))
        if recorded_source != source_signature:
            raise ValueError('Fitted source observations or native labels changed')
        if (trained_bundle['fitting'].get('algorithm') != 'paired_supervision'
                or trained_bundle['fitting'].get('seed') != seed
                or trained_bundle['fitting'].get('variant') != variant):
            raise ValueError('Fitted Controller algorithm, seed or variant changed')
        for name in ('dev_seed_predictions', 'dev_policy_seed_predictions', 'dev_ega_predictions'):
            _validate_source_predictions(trained_bundle.get(name, []), dev, name)
    elif fitted_origin is not None:
        raise ValueError('A fitted checkpoint declaration requires its complete trained bundle')
    if frozen_components is not None:
        if selected is not None:
            raise ValueError('Frontend reuse requires source-development selection')
        if not frontend_origin or not frontend_origin.get('checkpoint_sha256'):
            raise ValueError('Frontend reuse requires a hashed source checkpoint declaration')
        component_ids = frozen_component_identity(frozen_components, backbone.spec, train, dev,
                                                  original_reader_train, original_reader_dev, track, config, seed,
                                                  reader_refit=reader_refit)
        if pair_cache is not None and (not teacher_origin or teacher_origin.get('components') != component_ids
                or teacher_origin.get('source_records') != grpo.signature((_record_signature(train), _record_signature(dev)))
                or not teacher_origin.get('files')):
            raise ValueError('Frozen frontend targets require their verified teacher origin')
    if teacher_origin is not None and (frozen_components is None or pair_cache is None):
        raise ValueError('Teacher reuse requires both its frozen frontend and validated target cache')
    # Canonicalize derived reservations before hashing, including reader supplements.
    indexed = {id(rec): rec for rec in train+dev+reader_train+reader_dev}
    index_reader_positions(list(indexed.values()), config['reader'])
    request = (getattr(backbone, 'spec', {}), track, config, variant, seed, selected,
               _record_signature(train), _record_signature(dev),
               _record_signature(reader_train), _record_signature(reader_dev))
    if frozen_components is not None:
        request += (frontend_origin, component_ids)
    if reader_source_declaration is not None:
        request += (reader_source_declaration,
                    grpo.signature((_record_signature(original_reader_train), _record_signature(original_reader_dev))))
    if plain_teacher_origin is not None:
        request += (plain_teacher_origin,)
    if teacher_origin is not None:
        request += (teacher_origin,)
    if trained_bundle is not None:
        request += (fitted_origin, component_ids)
    expected = grpo.signature(request)
    request_signature = expected
    if directory is not None:
        declaration = directory/'fit_protocol.json'
        if declaration.exists():
            saved_protocol = json.loads(declaration.read_text(encoding='utf-8'))
            if saved_protocol.get('request_signature', saved_protocol['signature']) != request_signature:
                raise ValueError('Saved EGA stages belong to a different configuration or cohort')
            final_path = directory/'final.pt'
            if final_path.is_file():
                bundle = torch.load(final_path, map_location='cpu', weights_only=False)
                if bundle.get('fit_signature') != saved_protocol['signature']:
                    raise ValueError('Final EGA checkpoint does not match this training request')
                if 'gain_loss' in saved_protocol and bundle.get('gain_loss', {'mode': 'raw_brier'}) != saved_protocol['gain_loss']:
                    raise ValueError('Final EGA checkpoint target loss does not match its resolved protocol')
                budget_policy = bundle_budget_policy(bundle, config['controller'])
                if budget_policy and saved_protocol.get('budget_profile') != budget_policy['budget_profile']:
                    raise ValueError('Final Controller budget profile does not match its resolved protocol')
                target_policy = bundle_target_policy(bundle, config['controller'])
                if target_policy and saved_protocol.get('target_profile') != target_policy['target_profile']:
                    raise ValueError('Final Controller target scale does not match its resolved protocol')
                return restore_bundle(backbone, bundle), bundle, (None, None, None)
            if not resume:
                raise ValueError('Partial EGA run exists; pass --resume to continue it')
        else:
            write_json(declaration, {'schema': 'ega-fit-v1', 'signature': expected,
                                      'request_signature': request_signature,
                                      'frontend_origin': frontend_origin, 'frontend_components': component_ids,
                                      'teacher_origin': teacher_origin, 'fitted_origin': fitted_origin,
                                      'plain_teacher_origin': plain_teacher_origin})
    if trained_bundle is not None:
        controller = restore_bundle(backbone, trained_bundle)
        budget_policy = bundle_budget_policy(trained_bundle, config['controller'])
        target_policy = bundle_target_policy(trained_bundle, config['controller'])
        if budget_policy:
            expected = grpo.signature((expected, budget_policy))
            if directory:
                saved = json.loads((directory/'fit_protocol.json').read_text(encoding='utf-8'))
                saved.update(signature=expected, **budget_policy)
                write_json(directory/'fit_protocol.json', saved)
        if target_policy:
            expected = grpo.signature((expected, target_policy))
            if directory:
                saved = json.loads((directory/'fit_protocol.json').read_text(encoding='utf-8'))
                saved.update(signature=expected, **target_policy)
                write_json(directory/'fit_protocol.json', saved)
        plain = trained_bundle['dev_seed_predictions']
        enhanced = trained_bundle['dev_ega_predictions']
        cut, report = shared_source_threshold(plain, enhanced, 'seed_shared_source_dev')
        report['reused_source_predictions'] = True
        bundle = dict(trained_bundle, threshold=cut, calibration=report, fit_signature=expected,
                      fitted_origin=fitted_origin, source_records_signature=source_signature,
                      training_order=['restore_trained_bundle', 'source_plain_threshold'],
                      offline_seconds=time.perf_counter()-started)
        if directory:
            grpo.save_checkpoint(directory/'final.pt', bundle)
        print('recalibrated_fitted_bundle: '+json.dumps(report), flush=True)
        return controller, bundle, (None, None, None)
    adapter = None
    reader_bundle = None
    discriminator_bundle = None
    reader_refit_provenance = (frozen_components.get('reader_refit_provenance')
                               if frozen_components is not None and not reader_refit else None)
    if frozen_components is not None:
        restore_frontend(backbone, frozen_components, config['controller'])
        reader_bundle = frozen_components['reader']
        discriminator_bundle = frozen_components['discriminator']
        adapter = frozen_components.get('adapter')
        print('restored_frontend: '+json.dumps(component_ids), flush=True)
        if reader_refit:
            original_reader = discriminator_bundle['reader_fingerprint']
            backbone.model.eval().requires_grad_(False)
            reader_bundle = fit_reader(backbone, reader_train, reader_dev, config['reader'], seed,
                                       checkpoint_path=directory/'reader_training.pt' if directory else None,
                                       resume=resume)
            reader_refit_provenance = {
                'version': 1, 'original_reader': original_reader,
                'reader': reader_bundle['fingerprint'],
                'discriminator': discriminator_bundle['fingerprint'],
                'source_records': grpo.signature((_record_signature(reader_train), _record_signature(reader_dev))),
                'selection': 'source_train_and_dev_only', 'new_discriminator_training': False,
                'frontend_checkpoint_sha256': frontend_origin['checkpoint_sha256'],
            }
            if reader_source_declaration is not None:
                reader_refit_provenance.update(
                    source_declaration=reader_source_declaration,
                    refit_input_reader=frozen_components['reader']['fingerprint'],
                    original_source_records=source_signature,
                    original_reader_participants={'train': _participant_roster(train),
                                                  'dev': _participant_roster(dev)},
                    refit_input_reader_participants={'train': _participant_roster(original_reader_train),
                                                    'dev': _participant_roster(original_reader_dev)},
                    reader_participants={'train': _participant_roster(reader_train),
                                         'dev': _participant_roster(reader_dev)})
            reader_refit_provenance['fingerprint'] = digest(reader_refit_provenance)
            component_ids = dict(component_ids, reader=reader_bundle['fingerprint'])
            if directory:
                saved = json.loads((directory/'fit_protocol.json').read_text(encoding='utf-8'))
                saved.update(frontend_components=component_ids, reader_refit_provenance=reader_refit_provenance)
                write_json(directory/'fit_protocol.json', saved)
            print('reader_refit: '+json.dumps(reader_refit_provenance), flush=True)
    elif pair_cache is None:
        # Every fold starts at the foundation weights, never the previous fold's LoRA.
        if hasattr(backbone, 'reset_discriminator'):
            backbone.reset_discriminator()
        elif discriminator_enabled:
            raise ValueError('The backbone does not support discriminator LoRA reset')
        backbone.reader = None
        backbone.adapter = None
        backbone.adapter_bundle = None
        backbone.discriminator_bundle = None
    if track == 'F' and pair_cache is None and frozen_components is None:
        adapter = fit_adapter(backbone, train, dev, config['adapter'], selected.get('adapter') if selected else None,
                              checkpoint_path=directory/'adapter_training.pt' if directory else None,
                              resume=resume)
        backbone.adapter_bundle = adapter
    if pair_cache is None and frozen_components is None:
        reader_bundle = fit_reader(backbone, reader_train, reader_dev, config['reader'], seed,
                                   selected.get('reader') if selected else None,
                                   checkpoint_path=directory/'reader_training.pt' if directory else None,
                                   resume=resume)
        if discriminator_enabled:
            from ..discriminator import train_discriminator
            # Reader representations remain valid: base embeddings are frozen,
            # and LoRA trains language layers rather than source embeddings.
            backbone.reader.model.eval().requires_grad_(False)
            if getattr(backbone, 'adapter', None) is not None:
                backbone.adapter.eval().requires_grad_(False)
            discriminator_bundle = train_discriminator(
                backbone, train, dev, track, config['discriminator'],
                checkpoint_dir=directory/'discriminator' if directory else None,
                resume=resume, seed=seed,
                selected=selected.get('discriminator') if selected else None)
            if not discriminator_bundle.get('fingerprint'):
                raise ValueError('Trained discriminator is missing its checkpoint fingerprint')
            backbone.discriminator_bundle = discriminator_bundle
            backbone.model.eval().requires_grad_(False)
            # Rebind to the trained model wrapper while retaining the same frozen
            # base-embedding specification and reviewed Reader checkpoint.
            backbone.reader = ReaderRuntime(backbone, reader_bundle,
                                           include_relation_probabilities=evidence_encoding == 'structured_soft')
    elif pair_cache is not None:
        if getattr(backbone, 'reader', None) is None:
            raise ValueError('Cached teacher targets require the original frozen reader')
        reader_bundle = backbone.reader.bundle
        adapter = getattr(backbone, 'adapter_bundle', None)
        if track == 'F' and not adapter:
            raise ValueError('Cached feature targets require their fitted input adapter checkpoint')
        discriminator_bundle = getattr(backbone, 'discriminator_bundle', None)
        if discriminator_enabled and not discriminator_bundle:
            raise ValueError('Cached targets require their trained frozen discriminator')
        if any(e.get('reader_fingerprint') != backbone.reader.fingerprint for split in pair_cache[:2]
               for e in (split or [])):
            raise ValueError('Reader changed: recompute paired controller targets')
        if any(e.get('discriminator_fingerprint') != _discriminator_fingerprint(backbone)
               for split in pair_cache[:2] for e in (split or [])):
            raise ValueError('Discriminator changed: recompute paired controller targets')
        for field, default in (() if (teacher_origin or {}).get('input_rebuild_required') else
                               (('state_representation', 'hash'), ('seed_prediction_anchor', 'legacy_reader'))):
            if any(e.get(field, default) != config['controller'].get(field, default)
                   for split in pair_cache[:2] for e in (split or [])):
                raise ValueError(f'{field} changed: recompute paired controller targets')
        if not (teacher_origin or {}).get('state_evidence_required') and any(
                e.get('state_evidence', 'gated') != config['controller'].get('state_evidence', 'gated')
                for split in pair_cache[:2] for e in (split or [])):
            raise ValueError('Controller state evidence changed: re-encode source teacher states')
        if not (teacher_origin or {}).get('candidate_conditioning_required') and any(
                candidate_conditioning_mode(e) != candidate_mode
                for split in pair_cache[:2] for e in (split or [])):
            raise ValueError('Controller candidate conditioning changed: derive matching source masks')
        if config['controller'].get('state_representation') == 'frozen_token_embeddings' and any(
                e.get('state_representation_version') != STATE_REPRESENTATION_VERSION
                for split in pair_cache[:2] for e in (split or [])):
            raise ValueError('Frozen Controller representation changed: recompute paired controller targets')
    if evidence_encoding == 'structured_soft' and not getattr(
            backbone.reader, 'include_relation_probabilities', False):
        backbone.reader = ReaderRuntime(backbone, reader_bundle, include_relation_probabilities=True)
    calibration_config = config.get('calibration', {})
    complete_calibration = calibration_config.get('mode') == 'joint_source_dev'
    reader_gate_loss = calibration_config.get('reader_gate_loss', 'raw_brier')
    if reader_gate_loss not in ('raw_brier', 'controller_gain_loss'):
        raise ValueError('Unsupported source Reader gate loss')
    if reader_gate_loss == 'controller_gain_loss' and not complete_calibration:
        raise ValueError('Controller-unit Reader gate requires joint source development calibration')
    if (reader_gate_loss == 'controller_gain_loss' and directory is not None and resume
            and frozen_components is not None and frontend_origin):
        source = Path(frontend_origin['checkpoint']).parent
        base = (backbone.spec, _discriminator_fingerprint(backbone), track, _record_signature(dev))
        for name, schema, identity in (
                ('calibration.pt', 'ega-calibration-v1', base),
                ('reader_calibration.pt', 'ega-reader-calibration-v1',
                 (base[0], backbone.reader.fingerprint, *base[1:]))):
            if reader_refit and name == 'reader_calibration.pt':
                continue  # The new Reader requires fresh source predictions and gate selection.
            destination = directory/name
            if not destination.exists() and (source/name).is_file():
                cached = grpo.load_checkpoint(source/name, True, schema, grpo.signature(identity))
                _validate_source_predictions(cached['predictions'], dev, 'frozen frontend '+name)
                grpo.save_checkpoint(destination, cached)
    gate = None
    plain = None
    target_loss_provenance = None
    requested_loss = config['controller'].get('gain_loss', {})
    # Resolve the source-only cutoff before accepting/rejecting Reader evidence.
    # The same ordinary sample loss then governs the gate and all paired targets.
    if requested_loss.get('mode', 'raw_brier') == 'source_cut_centered_squared':
        if selected is not None or not dev:
            raise ValueError('Source-cut-centered targets require the frozen frontend source-development cohort')
        plain = teacher_origin.get('source_plain_predictions') if teacher_origin is not None else None
        if plain is None:
            plain = (_cached_seed_predictions(backbone, dev, track, directory/'calibration.pt', resume)
                     if directory else seed_predictions(backbone, dev, track))
        _validate_source_predictions(plain, dev, 'gain-loss source plain predictions')
        valid_plain = [{k: v for k, v in r.items() if k != 'threshold'} for r in plain
                       if r['status'] == 'ok' and r.get('probability_valid', True)]
        if not valid_plain:
            raise ValueError('Source-cut-centered targets require valid source plain probabilities')
        source_cut = threshold(valid_plain)
        if teacher_origin is not None and requested_loss.get('source_cut', source_cut) != source_cut:
            raise ValueError('Reused source plain predictions changed the recorded teacher cutoff')
        config = dict(config, controller=dict(config['controller'], gain_loss={
            'mode': 'source_cut_centered_squared', 'source_cut': source_cut}))
        target_loss_provenance = {'selection': 'plain_seed_source_dev_macro_f1', 'target_adaptation': False,
                                  'source_n': len(dev), 'selection_n': len(valid_plain),
                                  'source_prediction_signature': grpo.signature(valid_plain)}
    from ..controller import gain_loss_settings
    target_loss = gain_loss_settings(config['controller'])
    gate_profile = target_loss if reader_gate_loss == 'controller_gain_loss' else {'mode': 'raw_brier'}
    if complete_calibration:
        if teacher_origin is not None:
            gate = teacher_origin['reader_gate']
            if plain is None:
                plain = teacher_origin.get('source_plain_predictions')
            origin_generation = dict(teacher_origin['generation_config'])
            origin_generation.setdefault('teacher_sampling', {'mode': 'modality_uniform'})
            origin_generation.setdefault('teacher_input_rebuild', {'mode': 'disabled'})
            origin_generation.setdefault('candidate_conditioning', 'none')
            origin_generation.setdefault('teacher_rollout', {'mode': 'sampled_prefixes'})
            if teacher_origin.get('input_rebuild_required'):
                origin_generation = teacher_origin['rebuild_generation_config']
            requested_generation = teacher_generation_config(dict(config['controller'], reader_gate=gate),
                                                               include_augmentation=False)
            origin_generation = {k: v for k, v in origin_generation.items() if k != 'teacher_augmentation'}
            if teacher_origin.get('score_calibration_required'):
                requested_generation.pop('score_calibration')
                origin_generation.pop('score_calibration')
            if teacher_origin.get('state_evidence_required'):
                requested_generation.pop('state_evidence')
                origin_generation.pop('state_evidence')
            if teacher_origin.get('candidate_conditioning_required'):
                requested_generation.pop('candidate_conditioning')
                origin_generation.pop('candidate_conditioning')
            if requested_generation != origin_generation:
                raise ValueError('Teacher generation configuration changed before fitting')
        elif selected is not None:
            gate = selected.get('reader_gate')
            if gate is None:
                raise ValueError('Refitting requires the source-development Reader gate')
        else:
            if not dev:
                raise ValueError('Source development is required for full-chain calibration')
            if plain is None:
                plain = (_cached_seed_predictions(backbone, dev, track, directory/'calibration.pt', resume)
                         if directory else seed_predictions(backbone, dev, track))
            enhanced = _cached_reader_predictions(backbone, dev, track,
                                                  directory/'reader_calibration.pt' if directory else None, resume)
            gate = fit_reader_gate(plain, enhanced, calibration_config.get('reader_minimum_brier_gain', 0.),
                                   gain_loss=gate_profile)
        if gain_loss_settings({'gain_loss': gate.get('gain_loss', {})}) != gate_profile:
            raise ValueError('Reader gate loss changed: regenerate paired targets under the new gate')
        config = dict(config, controller=dict(config['controller'], reader_gate=gate))
        print('reader_gate: '+json.dumps(gate), flush=True)
    score_profile, score_report = {'mode': 'disabled'}, None
    fresh_audio_calibration = score_request['mode'] != 'disabled' and teacher_origin is None and pair_cache is None
    if score_request['mode'] != 'disabled':
        if fresh_audio_calibration:
            if (directory is None or selected is not None
                    or config['controller'].get('seed_prediction_anchor', 'legacy_reader') != 'plain_until_acquisition'):
                raise ValueError('Fresh Audio transport requires saved source teachers and a plain seed anchor')
            # The unresolved request cannot score native inputs. Persist raw
            # triplets once; resolve the source-only transport after generation.
            config = dict(config, controller=dict(config['controller'], score_calibration=score_profile))
        elif not teacher_origin or pair_cache is None:
            raise ValueError('Audio transport requires the verified completed source teacher checkpoint')
        elif teacher_origin.get('score_calibration_required'):
            score_profile, score_report = fit_source_audio(dev, pair_cache[1], track, plain,
                target_loss['source_cut'], component_ids, source_signature)
            if 'fingerprint' in score_request and score_request != score_profile:
                raise ValueError('Requested score calibration changed its source-only fitted profile')
        elif not fresh_audio_calibration:
            score_profile = validate_profile(config['controller']['score_calibration'])
        config = dict(config, controller=dict(config['controller'], score_calibration=score_profile))
        print('score_calibration: '+json.dumps(score_profile), flush=True)
    if target_loss_provenance is not None:
        target_loss_provenance['gain_loss'] = target_loss
        # Resume can validate the request before scoring; final identity also
        # binds the resolved source-only cutoff and its prediction cohort.
        expected = grpo.signature((request_signature, target_loss, target_loss_provenance))
    if reader_refit_provenance is not None:
        expected = grpo.signature((expected, reader_refit_provenance))
    if directory:
        declaration = directory/'fit_protocol.json'
        saved = json.loads(declaration.read_text(encoding='utf-8'))
        if saved.get('gain_loss', target_loss) != target_loss:
            raise ValueError('Source cutoff or target loss changed within this saved fit request')
        saved.update(signature=expected, request_signature=request_signature,
                     gain_loss=target_loss, target_loss_provenance=target_loss_provenance,
                     teacher_config=config['controller'], score_calibration=score_profile,
                     score_calibration_report=score_report)
        write_json(declaration, saved)
    print('teacher_target_loss: '+json.dumps(target_loss), flush=True)
    plain_teacher_provenance = None
    if pair_cache is None:
        if directory:
            teacher_directory = directory/('teacher_raw' if fresh_audio_calibration else 'teacher')
            if plain_teacher_origin is not None:
                _validate_plain_teacher_origin(backbone, train, dev, track, config, plain_teacher_origin)
                (train_pairs, dev_pairs, timings), plain_teacher_provenance = _refit_plain_teacher_pairs(
                    backbone, train, dev, track, config['controller'], plain_teacher_origin, teacher_directory, resume)
                expected = grpo.signature((expected, plain_teacher_provenance))
                declaration = directory/'fit_protocol.json'
                saved = json.loads(declaration.read_text(encoding='utf-8'))
                if (saved.get('plain_teacher_provenance') is not None
                        and saved['plain_teacher_provenance'] != plain_teacher_provenance):
                    raise ValueError('Saved Reader-refit plain Teacher derivation changed')
                saved.update(signature=expected, plain_teacher_origin=plain_teacher_origin,
                             plain_teacher_provenance=plain_teacher_provenance)
                write_json(declaration, saved)
            else:
                train_pairs, timings = _cached_teacher_pairs(backbone, train, track, config['controller'],
                                                             teacher_directory/'train', resume)
                dev_pairs = _cached_teacher_pairs(backbone, dev, track, config['controller'],
                                                   teacher_directory/'dev', resume)[0] if dev else None
        else:
            train_pairs, timings = teacher_pairs(backbone, train, track, config['controller'])
            dev_pairs = teacher_pairs(backbone, dev, track, config['controller'])[0] if dev else None
            _mark_discriminator_targets(train_pairs, backbone)
            _mark_discriminator_targets(dev_pairs or [], backbone)
    else:
        train_pairs, dev_pairs, timings = pair_cache
        expected_gate = reader_gate_signature(config['controller'].get('reader_gate'))
        if any(e.get('reader_gate_signature') != expected_gate
               for split in (train_pairs, dev_pairs or []) for e in split):
            raise ValueError('Reader gate changed: recompute paired controller targets')
    score_provenance = None
    if fresh_audio_calibration:
        score_profile, score_report = fit_source_audio(dev, dev_pairs or [], track, plain,
            target_loss['source_cut'], component_ids, source_signature)
        if 'fingerprint' in score_request and score_request != score_profile:
            raise ValueError('Requested score calibration changed its fresh source-only fitted profile')
        config = dict(config, controller=dict(config['controller'], score_calibration=score_profile))
        (train_pairs, dev_pairs, timings), score_provenance = _calibrate_cached_targets(
            backbone, train, dev, track, config['controller'], (train_pairs, dev_pairs, timings),
            None, directory, resume, raw_directory=directory/'teacher_raw')
        expected = grpo.signature((expected, score_profile, score_provenance))
        saved = json.loads((directory/'fit_protocol.json').read_text(encoding='utf-8'))
        if (saved.get('score_calibration_provenance') is not None
                and saved['score_calibration_provenance'] != score_provenance):
            raise ValueError('Saved fresh score-calibrated source targets changed')
        saved.update(signature=expected, teacher_config=config['controller'], score_calibration=score_profile,
                     score_calibration_report=score_report, score_calibration_provenance=score_provenance)
        write_json(directory/'fit_protocol.json', saved)
        print('score_calibration: '+json.dumps(score_profile), flush=True)
    elif teacher_origin is not None and teacher_origin.get('score_calibration_required'):
        (train_pairs, dev_pairs, timings), score_provenance = _calibrate_cached_targets(
            backbone, train, dev, track, config['controller'], (train_pairs, dev_pairs, timings),
            teacher_origin, directory, resume)
        expected = grpo.signature((expected, score_profile, score_provenance))
        declaration = directory/'fit_protocol.json'
        saved = json.loads(declaration.read_text(encoding='utf-8'))
        if (saved.get('score_calibration_provenance') is not None
                and saved['score_calibration_provenance'] != score_provenance):
            raise ValueError('Saved score-calibrated source targets changed')
        saved.update(signature=expected, score_calibration_provenance=score_provenance)
        write_json(declaration, saved)
    rebuild_provenance = None
    if teacher_origin is not None and teacher_origin.get('input_rebuild_required'):
        (train_pairs, dev_pairs, timings), rebuild_provenance = _rebuild_cached_targets(
            backbone, train, dev, track, config['controller'], (train_pairs, dev_pairs, timings),
            teacher_origin, directory, resume)
        expected = grpo.signature((expected, rebuild_provenance))
        declaration = directory/'fit_protocol.json'
        saved = json.loads(declaration.read_text(encoding='utf-8'))
        if (saved.get('teacher_input_rebuild_provenance') is not None
                and saved['teacher_input_rebuild_provenance'] != rebuild_provenance):
            raise ValueError('Saved derived teacher targets changed within this fit request')
        saved.update(signature=expected, teacher_input_rebuild_provenance=rebuild_provenance)
        write_json(declaration, saved)
    state_evidence_provenance = None
    if teacher_origin is not None and teacher_origin.get('state_evidence_required'):
        (train_pairs, dev_pairs, timings), state_evidence_provenance = _observed_reader_cached_targets(
            backbone, train, dev, track, config['controller'], (train_pairs, dev_pairs, timings),
            teacher_origin, directory, resume)
        expected = grpo.signature((expected, state_evidence_provenance))
        declaration = directory/'fit_protocol.json'
        saved = json.loads(declaration.read_text(encoding='utf-8'))
        if (saved.get('state_evidence_provenance') is not None
                and saved['state_evidence_provenance'] != state_evidence_provenance):
            raise ValueError('Saved observed Reader state inputs changed within this fit request')
        saved.update(signature=expected, state_evidence_provenance=state_evidence_provenance)
        write_json(declaration, saved)
    candidate_provenance = None
    if teacher_origin is not None and teacher_origin.get('candidate_conditioning_required'):
        (train_pairs, dev_pairs, timings), candidate_provenance = _candidate_conditioned_cached_targets(
            backbone, train, dev, track, config['controller'], (train_pairs, dev_pairs, timings),
            teacher_origin, directory, resume)
        expected = grpo.signature((expected, candidate_provenance))
        declaration = directory/'fit_protocol.json'
        saved = json.loads(declaration.read_text(encoding='utf-8'))
        if (saved.get('candidate_conditioning_provenance') is not None
                and saved['candidate_conditioning_provenance'] != candidate_provenance):
            raise ValueError('Saved candidate-conditioned source inputs changed within this fit request')
        saved.update(signature=expected, candidate_conditioning_provenance=candidate_provenance)
        write_json(declaration, saved)
    from ..controller import teacher_augmentation_settings
    augmentation = teacher_augmentation_settings(config['controller'])
    augmentation_provenance = None
    if augmentation['mode'] != 'disabled':
        (train_pairs, dev_pairs, timings), augmentation_provenance = _augment_cached_targets(
            backbone, train, dev, track, config['controller'], (train_pairs, dev_pairs, timings),
            teacher_origin, directory, resume)
        expected = grpo.signature((expected, augmentation_provenance))
        declaration = directory/'fit_protocol.json'
        saved = json.loads(declaration.read_text(encoding='utf-8'))
        if (saved.get('teacher_augmentation_provenance') is not None
                and saved['teacher_augmentation_provenance'] != augmentation_provenance):
            raise ValueError('Saved teacher augmentation targets changed within this fit request')
        saved.update(signature=expected, teacher_augmentation=augmentation,
                     teacher_augmentation_provenance=augmentation_provenance)
        write_json(declaration, saved)
    if not train_pairs or not timings:
        raise ValueError('No usable controller targets/timings; inspect persisted teacher exclusions')
    evidence_encoding_report = None
    if evidence_encoding in ('structured', 'structured_soft'):
        from .controller_evidence import derive_reader_rows
        encoded_splits, counts = [], {}
        for split, records, pairs in (('train', train, train_pairs), ('dev', dev, dev_pairs or [])):
            encoded = []
            for record in records:
                current_pairs = [e for e in pairs if e['participant'] == record['family']+':'+record['participant']
                                 and str(e['state_id'][0]) == str(record['id'])]
                encoded.extend(derive_reader_rows(record, config['controller'], current_pairs,
                                                  reader_bundle['config']['ontology'],
                                                  speaker_normalization=reader_bundle['config'].get(
                                                      'speaker_normalization', 'literal')))
            if len(encoded) != len(pairs):
                raise ValueError('Structured Reader migration did not cover the exact source teacher pairs')
            encoded_splits.append(encoded)
            counts[split] = {'pairs': len(encoded), 'reader_rows': sum(
                len(e.get('reader_evidence', e.get('state_reader_records', []))) for e in encoded)}
        train_pairs, dev_pairs = encoded_splits
        evidence_encoding_report = {'mode': evidence_encoding, 'source_only': True,
                                    'new_reader_calls': 0, 'new_llm_calls': 0, 'splits': counts}
        print('reader_state_encoding: '+json.dumps(evidence_encoding_report), flush=True)
    state_filter_report = None
    if state_filter != 'none':
        marked_splits, counts = [], {}
        for split, records, pairs in (('train', train, train_pairs), ('dev', dev, dev_pairs or [])):
            marked, record_counts = [], []
            for record in records:
                current_pairs = [e for e in pairs if e['participant'] == record['family']+':'+record['participant']
                                 and str(e['state_id'][0]) == str(record['id'])]
                record_pairs, record_summary = mark_record_states(record, current_pairs, state_filter)
                marked.extend(record_pairs)
                record_counts.append(record_summary)
            if len(marked) != len(pairs):
                raise ValueError('Controller state eligibility did not cover the exact source teacher pairs')
            marked_splits.append(marked)
            counts[split] = {key: sum(item[key] for item in record_counts) for key in
                            ('original_pairs', 'retained_pairs', 'removed_pairs',
                             'original_states', 'retained_states', 'removed_states')}
        train_pairs, dev_pairs = marked_splits
        state_filter_report = {'mode': state_filter, 'source_only': True, 'metadata_only': True,
                               'new_reader_calls': 0, 'new_llm_calls': 0, 'splits': counts}
        print('controller_state_filter: '+json.dumps(state_filter_report), flush=True)
    controller_train, controller_dev = train_pairs, dev_pairs
    if not config['controller'].get('grpo', {}).get('enabled', True):
        # Malformed generated answers remain auditable teacher outcomes, but
        # paired regression requires genuine matched probability targets.
        controller_train = [e for e in train_pairs if e.get('supervised_valid', True)]
        controller_dev = [e for e in (dev_pairs or []) if e.get('supervised_valid', True)]
        if not controller_train:
            raise ValueError('No valid paired training targets for the supervised controller')
        if selected is None and dev and not controller_dev:
            raise ValueError('No valid paired development targets for controller checkpoint selection')
    costs = fit_cost(timings)
    controller_config = dict(config['controller'], cost_model=costs)
    controller, fitting = fit(controller_train, controller_dev, controller_config, variant, seed,
                              selected.get('controller') if selected else None,
                              checkpoint_path=directory/'controller_training.pt' if directory else None,
                              resume=resume)
    from .budget_units import budget_normalization_mode
    budget_mode = budget_normalization_mode(fitting)
    budget_policy = {}
    if budget_mode != 'per_seed_wall':
        budget_policy = {'budget_normalization': budget_mode, 'budget_profile': fitting['budget_profile']}
        controller_config.update(budget_policy)
        config = dict(config, controller=dict(config['controller'], **budget_policy))
        expected = grpo.signature((expected, budget_policy))
        if directory:
            declaration = directory/'fit_protocol.json'
            saved = json.loads(declaration.read_text(encoding='utf-8'))
            if saved.get('budget_profile') is not None and saved['budget_profile'] != fitting['budget_profile']:
                raise ValueError('Saved Controller source budget unit differs from this training cohort')
            saved.update(signature=expected, **budget_policy)
            write_json(declaration, saved)
    from ..controller import target_normalization_mode, validate_target_profile
    target_mode = target_normalization_mode(fitting)
    target_policy = {}
    if target_mode != 'none':
        target_profile = validate_target_profile(fitting.get('target_profile'))
        if target_profile is None or target_profile['mode'] != target_mode:
            raise ValueError('Fitted Controller is missing its source target scale')
        target_policy = {'target_normalization': target_mode, 'target_profile': target_profile}
        controller_config.update(target_policy)
        config = dict(config, controller=dict(config['controller'], **target_policy))
        expected = grpo.signature((expected, target_policy))
        if directory:
            declaration = directory/'fit_protocol.json'
            saved = json.loads(declaration.read_text(encoding='utf-8'))
            if saved.get('target_profile') is not None and saved['target_profile'] != target_profile:
                raise ValueError('Saved Controller target scale differs from its source training cohort')
            saved.update(signature=expected, **target_policy)
            write_json(declaration, saved)
    fitting['target_loss_provenance'] = target_loss_provenance
    fitting['state_evidence_encoding'] = evidence_encoding
    if evidence_encoding == 'structured_soft':
        from .reader_state import reader_state_profile
        fitting['state_evidence_encoding_profile'] = reader_state_profile(config['controller'])
    fitting['state_evidence_encoding_provenance'] = evidence_encoding_report
    fitting['state_filter_provenance'] = state_filter_report
    if score_profile['mode'] != 'disabled':
        fitting['score_calibration'] = score_profile
        fitting['score_calibration_provenance'] = score_provenance
    if rebuild_provenance is not None:
        fitting['teacher_input_rebuild'] = rebuild_provenance
    if augmentation_provenance is not None:
        fitting['teacher_augmentation'] = augmentation_provenance
    if state_evidence_provenance is not None:
        fitting['state_evidence_provenance'] = state_evidence_provenance
    if candidate_provenance is not None:
        fitting['candidate_conditioning_provenance'] = candidate_provenance
    calibration = []
    if selected:
        policy = selected['policy'] | {'cost': costs} | budget_policy | target_policy
        policy.update({field: config['controller'].get(field, default) for field, default in
                       (('state_representation', 'hash'), ('seed_prediction_anchor', 'legacy_reader'),
                        ('state_evidence', 'gated'), ('candidate_conditioning', 'none'),
                        ('action_filter', 'none'), ('state_filter', 'none'), ('state_evidence_encoding', 'hash'),
                        ('reader_diagnostics', False))})
        cut = selected['threshold']
    else:
        if 'policy_lambda' in config['controller'] and 'policy_gamma' in config['controller']:
            policy = {'lambda': config['controller']['policy_lambda'],
                      'gamma': 0. if variant == 'no_presentation' else config['controller']['policy_gamma'], 'cost': costs,
                      'selection': 'explicit_launcher_weights'}
            policy.update(budget_policy)
            policy.update(target_policy)
        else:
            policy = select_policy(backbone, dev, track, controller, costs,
                                   config['controller'].get('budget', 8), 8, variant,
                                   policy_context=budget_policy | target_policy)
        policy.update({field: config['controller'].get(field, default) for field, default in
                       (('state_representation', 'hash'), ('seed_prediction_anchor', 'legacy_reader'),
                        ('state_evidence', 'gated'), ('candidate_conditioning', 'none'),
                        ('action_filter', 'none'), ('state_filter', 'none'), ('state_evidence_encoding', 'hash'),
                        ('reader_diagnostics', False))})
        if gate is not None:
            policy['reader_gate'] = gate
        policy['gain_loss'] = target_loss
        if score_profile['mode'] != 'disabled':
            policy['score_calibration'] = score_profile
        calibration = plain
        if calibration is None:
            calibration = (_cached_seed_predictions(backbone, dev, track, directory/'calibration.pt', resume)
                           if directory else seed_predictions(backbone, dev, track))
        valid_calibration = [r for r in calibration if r['status'] == 'ok' and r.get('probability_valid', True)]
        if not valid_calibration:
            raise ValueError('No valid unaugmented development predictions for threshold calibration')
        if complete_calibration:
            policy_seed, policy_ega = _cached_policy_predictions(
                backbone, dev, track, controller, policy, config['controller'],
                directory/'policy_calibration.pt' if directory else None, resume)
            cut, calibration_report = shared_source_threshold(
                policy_seed, policy_ega, calibration_config.get('threshold_objective', 'mean_seed_ega_macro_f1'))
        else:
            cut, calibration_report, policy_seed, policy_ega = threshold(valid_calibration), None, [], []
    if selected:
        calibration_report, policy_seed, policy_ega = None, [], []
        if gate is not None:
            policy['reader_gate'] = gate
        policy['gain_loss'] = target_loss
    if rebuild_profile['mode'] != 'disabled':
        policy['teacher_input_rebuild'] = rebuild_profile
    if score_profile['mode'] != 'disabled':
        policy['score_calibration'] = score_profile
    policy['state_evidence'] = config['controller'].get('state_evidence', 'gated')
    policy['state_evidence_encoding'] = evidence_encoding
    policy['reader_diagnostics'] = bool(config['controller'].get('reader_diagnostics', False))
    policy['candidate_conditioning'] = candidate_mode
    policy['action_filter'] = action_filter
    policy['state_filter'] = state_filter
    bundle = {'schema': 'ega-v2' if discriminator_enabled else 'ega-v1',
              'controller': controller.state_dict(), 'reader': reader_bundle, 'fitting': fitting, 'policy': policy,
              'threshold': cut, 'adapter': adapter, 'discriminator': discriminator_bundle,
              'training_order': (['restore_frozen_frontend'] if frozen_components is not None else
                                 ['feature_adapter']*int(track == 'F') + ['reader']
                                 + ['discriminator_grpo']*int(discriminator_enabled))
                                + ['source_reader_refit']*int(reader_refit)
                                + ['reuse_verified_paired_targets' if teacher_origin is not None else 'paired_targets',
                                    *(['source_audio_score_transport'] if score_provenance is not None else []),
                                    *(['observed_reader_controller_states'] if state_evidence_provenance is not None else []),
                                    *(['observed_event_candidate_conditioning'] if candidate_provenance is not None else []),
                                    *(['augment_aligned_audio_targets'] if augmentation_provenance is not None else []),
                                   'controller_grpo' if fitting['optimizer'] == 'grpo'
                                   else 'controller_paired_supervision'],
              'frontend_origin': frontend_origin, 'frontend_components': component_ids,
              'reader_refit_provenance': reader_refit_provenance,
              'plain_teacher_origin': plain_teacher_origin, 'plain_teacher_provenance': plain_teacher_provenance,
              'teacher_origin': teacher_origin,
              'teacher_augmentation': augmentation_provenance,
              'gain_loss': target_loss, 'target_loss_provenance': target_loss_provenance,
              'source_records_signature': source_signature,
              'offline_seconds': time.perf_counter()-started,
              'train_participants': sorted({r['family']+':'+r['participant'] for r in train}),
              'dev_participants': sorted({r['family']+':'+r['participant'] for r in dev}),
              'teacher_pair_count': len(train_pairs), 'dev_seed_predictions': calibration,
              'controller_pair_count': len(controller_train),
              'calibration': calibration_report, 'dev_policy_seed_predictions': policy_seed,
              'dev_ega_predictions': policy_ega,
              'teacher_input_rebuild': rebuild_provenance,
              'state_evidence_provenance': state_evidence_provenance,
              'state_evidence_encoding_provenance': evidence_encoding_report,
              'state_filter_provenance': state_filter_report,
              'candidate_conditioning_provenance': candidate_provenance,
              'score_calibration': score_profile, 'score_calibration_report': score_report,
              'score_calibration_provenance': score_provenance,
              'fit_signature': expected, 'fit_request_signature': request_signature}
    if budget_policy or target_policy:
        bundle['controller_config'] = controller_config
    if directory:
        grpo.save_checkpoint(directory/'final.pt', bundle)
    return controller, bundle, (train_pairs, dev_pairs, timings)


def restore_frontend(backbone, bundle, controller_config=None):
    """Restore foundation -> input adapter -> discriminator LoRA -> frozen Reader."""
    if bundle.get('schema') not in ('ega-v1', 'ega-v2') or not bundle.get('reader'):
        raise ValueError('EGA checkpoint is missing its evidence reader')
    if bundle['schema'] == 'ega-v2' and not bundle.get('discriminator'):
        raise ValueError('EGA checkpoint is missing its trained Screening discriminator')
    if hasattr(backbone, 'reset_discriminator'):
        backbone.reset_discriminator()
    elif bundle.get('discriminator'):
        raise ValueError('The backbone does not support discriminator LoRA reset')
    backbone.reader = None
    backbone.adapter = None
    backbone.adapter_bundle = None
    backbone.discriminator_bundle = None
    if bundle.get('adapter'):
        from .features import FeatureAdapter
        adapter = bundle['adapter']
        size = backbone.model.get_input_embeddings().weight.shape[1]
        backbone.adapter = FeatureAdapter(adapter['stats'], size, backbone.device, backbone.dtype)
        backbone.adapter.load_state_dict(adapter['state_dict'])
        backbone.adapter.eval().requires_grad_(False)
        backbone.adapter_bundle = adapter
    if bundle.get('discriminator'):
        from ..discriminator import restore_discriminator
        restore_discriminator(backbone, bundle['discriminator'])
        backbone.discriminator_bundle = bundle['discriminator']
        backbone.model.eval().requires_grad_(False)
    from .reader_state import reader_state_mode
    mode = reader_state_mode(bundle.get('policy', {}) if controller_config is None else controller_config)
    backbone.reader = ReaderRuntime(backbone, bundle['reader'],
                                   include_relation_probabilities=mode == 'structured_soft')


def restore_bundle(backbone, bundle):
    bundle_budget_policy(bundle)
    target_policy = bundle_target_policy(bundle)
    restore_frontend(backbone, bundle, bundle.get('policy', {}))
    controller = Controller.from_state_dict(bundle['controller'],
                                           target_profile=target_policy.get('target_profile'))
    controller.eval().requires_grad_(False)
    return controller


def evaluate_policy(backbone, records, track, controller, bundle, variant='full', limit=8):
    pre, post = [], []
    if bundle.get('reader'):
        index_reader_positions(records, bundle['reader']['config'])
    for rec in records:
        try:
            initial, final = acquire(backbone, rec, track, controller, bundle['policy'], 8, limit, variant)
        except ObservationError as exc:
            initial = final = {'p': .5, 'status': 'input_failure', 'error': str(exc), 'seconds': 0., 'gpu_seconds': 0.}
        pre.append(row(rec, initial, bundle['threshold']))
        post.append(row(rec, final, bundle['threshold']))
    return pre, post


def summary_pair(pre, post, cut):
    return {'seed': classification(pre, cut), 'ega': classification(post, cut),
            'difference_ci95': bootstrap(pre, post, cut), 'transitions': paired(pre, post, cut)}


def table4(backbone, records, task, config, out, seed):
    """Prespecified, independent frozen-backbone pairs with resumable predictions."""
    dev, test = ([r for r in records if r['split'] == s] for s in ('dev', 'test'))
    if not dev or not test:
        raise ValueError('Paired interventions require separate development and test participants')
    modes = ('field_layout', 'repeated_view', 'new_transcript', 'aligned_audio',
             'new_transcript_control', 'aligned_audio_control')
    plan = {}
    # Select every view before consulting predictions. Labels never enter selection.
    for rec in dev+test:
        selections = {}
        try:
            initial = seed_views(rec, 'T')
            seed_ids = [v['id'] for v in initial]
            selections['seed'] = {'view_ids': seed_ids, 'renderer': 0, 'repeat_id': None}
            if rec['split'] == 'test':
                selections['field_layout'] = selections['seed'] | {'renderer': 1}
                selections['repeated_view'] = selections['seed'] | {'renderer': 2, 'repeat_id': initial[0]['id']}
                choices = [v for v in pool(rec, 'T') if v['id'] not in seed_ids
                           and v.get('speaker', '').lower() == 'participant']
                after = [v for v in choices if v['start'] >= initial[1]['stop']]
                if choices:
                    new = (after or choices)[0]
                    selections['new_transcript'] = {'view_ids': seed_ids+[new['id']], 'renderer': 3,
                                                   'repeat_id': new['id']}
                    old = min(initial, key=lambda v: (abs(len(v.get('text', ''))-len(new.get('text', ''))),
                                                       abs(v['start']-new['start']), v['id']))
                    selections['new_transcript_control'] = selections['seed'] | {'renderer': 2, 'repeat_id': old['id']}
                else:
                    selections['new_transcript'] = selections['new_transcript_control'] = {'error': 'No unread eligible transcript'}
                audio = [v for v in pool(rec, 'A') if v['event'] == initial[1]['event']]
                if len(audio) == 1:
                    selections['aligned_audio'] = {'view_ids': seed_ids+[audio[0]['id']], 'renderer': 3,
                                                   'repeat_id': audio[0]['id']}
                    selections['aligned_audio_control'] = selections['seed'] | {'renderer': 2, 'repeat_id': initial[1]['id']}
                else:
                    selections['aligned_audio'] = selections['aligned_audio_control'] = {'error': 'Aligned audio unavailable'}
        except ObservationError as exc:
            selections = {mode: {'error': str(exc)} for mode in ('seed', *modes)}
        plan[rec['id']] = selections
    declaration = {'schema': 'paired-v2', 'seed': seed, 'selection': plan,
                   'rule': 'Three uniform participant transcript seeds; first unread transcript after middle seed (fallback earliest); audio aligned to middle seed.',
                   'controls': 'Append marked existing transcript in the same final presentation slot; transcript control minimizes character-length mismatch (token-length proxy), then time distance; audio control repeats the aligned transcript and is modality-unmatched.',
                   'bootstrap_repeats': 2000}
    plan_path, prediction_path = out/'pairs.json', out/'predictions.json'
    if plan_path.is_file():
        if json.loads(plan_path.read_text(encoding='utf-8')) != declaration:
            raise RuntimeError('Prespecified intervention plan changed; use a new run configuration')
    else:
        write_json(plan_path, declaration)
    outputs = json.loads(prediction_path.read_text(encoding='utf-8')) if prediction_path.is_file() else {}
    for mode, cohort in [('dev', dev), ('seed', test), *[(m, test) for m in modes]]:
        values = outputs.setdefault(mode, [])
        done = {r['id'] for r in values}
        if len(done) != len(values) or not done <= {r['id'] for r in cohort}:
            raise ValueError('Cached predictions do not match the declared cohort')
        for rec in cohort:
            if rec['id'] in done:
                continue
            if backbone is None:
                raise ValueError(f'Missing prediction for {mode}/{rec["id"]}; resume inference first')
            choice = plan[rec['id']]['seed' if mode == 'dev' else mode]
            try:
                if 'error' in choice:
                    raise ObservationError(choice['error'])
                by_id = {v['id']: v for v in rec['views']}
                state = [by_id[key] for key in choice['view_ids']]
                score = getattr(backbone, 'score_unaugmented', backbone.score)
                prediction = score(state, choice['renderer'], choice['repeat_id'])
                prediction['source_duration_seconds'] = sum(v['stop']-v['start'] for v in state)
                prediction['repeated_duration_seconds'] = (by_id[choice['repeat_id']]['stop']-by_id[choice['repeat_id']]['start']
                                                           if choice['renderer'] == 2 else 0.)
            except ObservationError as exc:
                prediction = {'p': .5, 'status': 'input_failure', 'error': str(exc), 'seconds': 0., 'gpu_seconds': 0.}
            values.append(row(rec, prediction) | {'presentation': choice})
            write_json(prediction_path, outputs)
            print(json.dumps({'intervention': mode, 'participant': rec['id'], 'done': len(values),
                              'total': len(cohort), 'status': prediction['status']}), flush=True)
    cut = threshold(outputs['dev'])
    pre = outputs['seed']
    result = {'schema': 'paired-v2', 'threshold': cut, 'threshold_source': 'unaugmented development seeds',
              'bootstrap': {'unit': 'participant', 'repeats': 2000, 'seed': seed, 'interval': 'percentile 95%'},
              'baseline': classification(pre, cut), 'development': classification(outputs['dev'], cut)}
    for mode in modes:
        values = outputs[mode]
        result[mode] = paired(pre, values, cut) | {
            'ci95': paired_bootstrap(pre, values, cut, seed=seed),
            'task_metrics': classification(values, cut),
            'task_difference_ci95': bootstrap(pre, values, cut, seed=seed)}
    layouts = [{r['id']: r['p'] for r in outputs[m]} for m in ('seed', 'field_layout', 'repeated_view')]
    variances = []
    for r in pre:
        probabilities = [layout[r['id']] for layout in layouts]
        mean = sum(probabilities)/len(probabilities)
        variances.append({'id': r['id'], 'score_variance': sum((p-mean)**2 for p in probabilities)/len(probabilities)})
    result['presentation_sensitivity'] = {
        'distribution': 'Uniform over canonical, field layout, and marked repeated view',
        'mean_score_variance': sum(r['score_variance'] for r in variances)/len(variances),
        'participants': variances}
    result['acquisition_vs_control'] = {}
    for mode in ('new_transcript', 'aligned_audio'):
        control = outputs[mode+'_control']
        treated = {r['id']: r for r in outputs[mode]}
        mismatch = []
        for r in control:
            z = treated[r['id']]
            def difference(key):
                return z[key]-r[key] if z.get(key) is not None and r.get(key) is not None else None
            mismatch.append({'id': r['id'], 'input_token_difference': difference('input_tokens'),
                             'text_token_difference': difference('text_tokens'),
                             'audio_token_difference': difference('audio_tokens'),
                             'media_seconds_difference': difference('media_seconds'),
                             'displayed_duration_difference': (z['source_duration_seconds']-r['source_duration_seconds']-r['repeated_duration_seconds']
                                 if 'source_duration_seconds' in z and 'source_duration_seconds' in r else None),
                             'modality_matched': mode == 'new_transcript'})
        result['acquisition_vs_control'][mode] = paired(control, outputs[mode], cut) | {
            'ci95': paired_bootstrap(control, outputs[mode], cut, seed=seed), 'mismatches': mismatch}
    return result


def requirements(root, config, jobs, table, stage):
    import transformers as tr
    errors = []
    for task_id, model_id in jobs:
        task, model = config['tasks'][task_id], config['models'][model_id]
        if stage != 'score' and not Path(model['path']).is_dir():
            errors.append(f'{model_id}: missing checkpoint {model["path"]}')
        if stage != 'score' and model['kind'] == 'omni3' and not hasattr(tr, 'Qwen3OmniMoeThinkerForConditionalGeneration'):
            errors.append('Q3O: installed Transformers lacks Qwen3OmniMoeThinkerForConditionalGeneration')
        try:
            records = load_records(root, task)
            if not any(r.get('split') == 'test' for r in records) and task.get('evaluation') != 'nested':
                errors.append(task_id+': no test records')
            if stage == 'train' and config.get('reader', {}).get('enabled', False):
                validate_config(config['reader'])
                annotations = load_annotations(root/config['reader']['annotations'])
                source = records
                splits = ('train', 'dev')
                for split in splits:
                    selected_rows = [r for r in source if r.get('split', 'pool') == split]
                    if not annotations_for(selected_rows, annotations, config['reader']):
                        errors.append(f'{task_id}: missing human reader annotations for {split}')
        except (ValueError, FileNotFoundError, KeyError) as exc:
            errors.append(task_id+': '+str(exc))
    return sorted(set(errors))


def prepare_gpu(requested):
    gpu = str(requested) if requested is not None else os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu or ',' in gpu:
        raise SystemExit('Select one GPU with --gpu INDEX or CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def export_csv(path, output):
    import csv
    rows = []
    def visit(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(prefix+[str(key)], child)
        else:
            rows.append(['/'.join(prefix), json.dumps(value, ensure_ascii=False)])
    visit([], output)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['task/model/seed/metric', 'value'])
        writer.writerows(rows)


def input_provenance(records, model):
    rows = [{k: v for k, v in r.items() if k != 'index_seconds'} for r in records]
    rows = [r | {'views': [{k:v for k,v in view.items() if k != 'reader_positions'} for view in r['views']]} for r in rows]
    paths = set()
    for rec in records:
        for view in rec['views']:
            if view.get('path'):
                paths.add(view['path'])
            paths.update(v['path'] for v in view.get('features', {}).values())
    media = {}
    for name in sorted(paths):
        path = Path(name)
        media[name] = {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns} if path.is_file() else None
    checkpoint = Path(model['path'])
    model_files = {p.name: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                   for p in checkpoint.glob('*') if p.is_file()}
    configs = {p.name: p.read_text(encoding='utf-8') for p in checkpoint.glob('*.json')}
    return {'records_sha256': digest(rows), 'media': media, 'model_files': model_files, 'model_configs_sha256': digest(configs)}


def main(table, defaults=None):
    if table != 4:
        raise ValueError('This experiment driver is not implemented for the current paper')
    parser = argparse.ArgumentParser(description='EGA training and paired evidence interventions')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--config', default='configs/experiments.json')
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--stage', choices=('run', 'collect', 'score', 'train'), default='run')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--reader-epochs', type=int)
    parser.add_argument('--reader-learning-rate', type=float)
    parser.add_argument('--reader-patience', type=int)
    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root/args.config).read_text(encoding='utf-8'))
    for name in ('epochs', 'learning_rate', 'patience'):
        value = getattr(args, name)
        if value is not None:
            if value <= 0:
                parser.error(f'{name} must be positive')
            config['controller'][name] = value
    for name in ('epochs', 'learning_rate', 'patience'):
        value = getattr(args, 'reader_'+name)
        if value is not None:
            if value <= 0:
                parser.error(f'reader_{name} must be positive')
            config['reader'][name] = value
    jobs = [tuple(j) for j in config['tables'][str(table)]]
    if args.models:
        if not set(args.models) <= {m for _, m in jobs}:
            parser.error('Model is not part of this paper table')
        jobs = [(t, m) for t, m in jobs if m in args.models]
    if args.tasks:
        if not set(args.tasks) <= {t for t, _ in jobs}:
            parser.error('Task is not part of this paper table')
        jobs = [(t, m) for t, m in jobs if t in args.tasks]
    errors = requirements(root, config, jobs, table, args.stage)
    if args.check:
        print(json.dumps({'table': table, 'jobs': jobs, 'ready': not errors, 'requirements': errors}, indent=2))
        return int(bool(errors))
    if errors:
        raise SystemExit('\n'.join(errors))
    if args.stage != 'train' and args.seeds and len(args.seeds) != 1:
        parser.error('Frozen paired interventions use one deterministic seed; controller seeds apply only to training')
    if args.stage != 'score':
        prepare_gpu(args.gpu)
    code = {p.relative_to(root).as_posix(): p.read_text(encoding='utf-8') for p in (root/'code/tmm').rglob('*.py')}
    annotation_path = root/config.get('reader', {}).get('annotations', 'manifests/reader/items.jsonl')
    annotation_hash = digest(load_annotations(annotation_path)) if annotation_path.is_file() else None
    fingerprint = digest({'config': config, 'code': code, 'reader_annotations': annotation_hash})
    if 'reader' in config:
        config['reader']['annotations'] = str(annotation_path.resolve())
    base = root/'runs'/('ega_train' if args.stage == 'train' else f'table{table}')/fingerprint[:12]
    base.mkdir(parents=True, exist_ok=True)
    write_json(base/'protocol.json', {'fingerprint': fingerprint, 'config': config, 'jobs': jobs})
    output = {}
    for task_id, model_id in jobs:
        spec = dict(config['tasks'][task_id])
        records = load_records(root, spec)
        seeds = args.seeds or (config['controller_seeds'] if args.stage == 'train' else [config['seed']])
        for seed in seeds:
            out = base/task_id/model_id/str(seed)
            out.mkdir(parents=True, exist_ok=True)
            provenance = input_provenance(records, config['models'][model_id])
            manifest = out/'inputs.json'
            if manifest.exists():
                previous = json.loads(manifest.read_text())
                if args.stage != 'score' and previous != provenance:
                    raise RuntimeError('Source or checkpoint changed; use a new run configuration')
            else:
                write_json(manifest, provenance)
            if (out/'metrics.json').is_file():
                existing = json.loads((out/'metrics.json').read_text())
                if existing.get('stage') != 'human_review' and args.stage != 'score':
                    output[f'{task_id}/{model_id}/{seed}'] = existing
                    continue
            if args.stage == 'score':
                result = table4(None, records, spec, config, out, seed)
            else:
                torch.manual_seed(seed)
                backbone = Backbone(config['models'][model_id], config['protocol'], spec['instruction'])
                if args.stage == 'train':
                    train = [r for r in records if r['split'] == 'train']
                    dev = [r for r in records if r['split'] == 'dev']
                    if not train or not dev:
                        raise ValueError('EGA training requires separate train and development participants')
                    controller, bundle, _ = fit_bundle(backbone, train, dev, spec['track'], config, seed=seed)
                    torch.save(bundle, out/'controller.pt')
                    result = {'stage': 'trained', 'checkpoint': str(out/'controller.pt'),
                              'fitting': bundle['fitting'], 'policy': bundle['policy'],
                              'threshold': bundle['threshold'], 'teacher_pair_count': bundle['teacher_pair_count']}
                    del controller
                else:
                    result = table4(backbone, records, spec, config, out, seed)
                del backbone
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            write_json(out/'metrics.json', result)
            output[f'{task_id}/{model_id}/{seed}'] = result
            print(json.dumps({'table': table, 'task': task_id, 'model': model_id, 'seed': seed, 'output': str(out)}), flush=True)
    write_json(base/'table.json', output)
    export_csv(base/'table.csv', output)
    return 0
