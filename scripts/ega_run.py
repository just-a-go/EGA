#!/usr/bin/env python3
"""Full EGA training/evaluation with participant isolation and resumable stages.

The frozen foundation model is adapted with token-level GRPO before controller fitting.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
sys.dont_write_bytecode = True


class BlockedError(ValueError):
    """A source, protocol, or calibration prerequisite is unavailable."""


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def people(records):
    return {(r['family'], r['participant']) for r in records}


def repartition(records, split):
    # reader_positions is added during fitting; each fold owns its own views.
    return [dict(copy.deepcopy(r), split=split) for r in records]


def merge_config(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_config(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def fitting_settings(settings):
    """Calibration-only reuse permits one cutoff objective change, no training changes."""
    value = copy.deepcopy(settings)
    for key in ('stage', 'gpu', 'parameter_provenance', 'fitted_checkpoint'):
        value.pop(key, None)
    value.get('calibration', {}).pop('threshold_objective', None)
    return value


def gate_calibration(settings):
    value = copy.deepcopy(settings.get('calibration', {}))
    value.pop('threshold_objective', None)
    return value


def split_annotations(annotations, records):
    """Copy fold partitions into selected rows; leave source annotations intact."""
    lookup = {(r['family'], r['id']): r for r in records}
    if len(lookup) != len(records):
        raise BlockedError('Reader sources contain duplicate family/record identities')
    rows = []
    for annotation in annotations:
        rec = lookup.get((annotation['family'], annotation['record_id']))
        if rec is None:
            continue
        if annotation['participant'] != rec['participant']:
            raise BlockedError('Reader annotation participant disagrees with its source')
        rows.append(dict(annotation, split=rec['split']))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows)
    if path.is_file() and path.read_text(encoding='utf-8') == content:
        return
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(content, encoding='utf-8')
    temp.replace(path)


def calibration_rows(bundle, dev):
    """Retain all dev outcomes in the bundle; fit the cutoff on valid outcomes."""
    rows = bundle.get('dev_seed_predictions', [])
    expected = {r['id'] for r in dev}
    if len(rows) != len(expected) or {r['id'] for r in rows} != expected:
        raise BlockedError('Development seed predictions do not cover the full declared cohort')
    valid = [r for r in rows if r.get('status') == 'ok' and r.get('probability_valid') is True]
    if not valid:
        raise BlockedError('No valid development seed probabilities for classifier calibration')
    return [{k: v for k, v in r.items() if k != 'threshold'} for r in valid]


def select_outer(bundles, dev_predictions, config):
    """Inner folds select duration, reader presence and the seed decision cutoff."""
    from tmm.support.metrics import threshold
    epoch_mean = lambda values: max(1, int(round(statistics.mean(values))))
    fitting = bundles[0]['fitting']
    selected = {
        'reader': {'epochs': epoch_mean([b['reader']['epochs'] for b in bundles]),
                   'threshold': statistics.median([b['reader']['threshold'] for b in bundles])},
        'controller': {'epochs': epoch_mean([b['fitting']['epochs'] for b in bundles]),
                       'alpha': fitting['alpha'], 'beta': fitting['beta']},
        'policy': {'lambda': config['controller']['policy_lambda'],
                   'gamma': config['controller']['policy_gamma']},
        'threshold': threshold(dev_predictions),
        'discriminator': {'epochs': epoch_mean([b['discriminator']['epochs'] for b in bundles])},
    }
    adapters = [b['adapter'] for b in bundles if b.get('adapter')]
    if adapters:
        if len(adapters) != len(bundles):
            raise BlockedError('Only some inner folds contain the feature adapter')
        selected['adapter'] = {'epochs': epoch_mean([a['epochs'] for a in adapters]),
                               'weight_decay': config['adapter']['weight_decay'][0]}
    return selected


class EgaRun:
    def __init__(self, settings, args):
        self.settings, self.args = settings, args
        self.seed = settings['seed']
        self.task_id, self.model_id = settings['task'], settings['model']
        self.stage = args.stage or settings.get('stage', 'all')
        self.config = load_json(ROOT / settings.get('config', 'configs/experiments.json'))
        for key in ('reader', 'discriminator', 'controller', 'adapter', 'protocol', 'calibration'):
            self.config.setdefault(key, {})
            merge_config(self.config[key], settings.get(key, {}))
        self.config['reader_refit'] = settings.get('reader_refit', False)
        self.config['controller']['seed'] = self.seed
        self.config['adapter']['seed'] = self.seed
        self.model = copy.deepcopy(self.config['models'][self.model_id])
        self.model['path'] = settings.get('pretrained_weights', self.model['path'])
        self.model['screening'] = copy.deepcopy(settings['screening'])
        self.model['disable_compile'] = settings.get('disable_compile', False)
        self.task = copy.deepcopy(self.config['tasks'][self.task_id])
        self.source_id = settings.get('training_task', self.task_id)
        self.source_task = copy.deepcopy(self.config['tasks'][self.source_id])
        self.evaluation = settings['evaluation']
        self.annotations_path = ROOT / self.config['reader']['annotations']
        self.config['reader']['annotations'] = str(self.annotations_path)
        self.backbone = None
        self.frontend_origin = None
        self.frozen_components = None
        self.teacher_origin = None
        self.teacher_declaration = None
        self.teacher_cache = None
        self.validated_teacher_origin = None
        self.plain_teacher_origin = None
        self.plain_teacher_verified = False
        self.fitted_origin = None
        self.reader_refit_origin = None
        self.reader_refit_source_ids = None
        self.restored_reader_source_declaration = None
        self.out = None
        self.output_bound = False
        self.check_training_blockers, self.check_evaluation_blockers = [], []
        self.require_full_settings()

    def require_full_settings(self):
        from tmm.support.reader_sources import reader_refit_sources
        from tmm.support.budget_units import budget_normalization_mode
        budget_normalization_mode(self.config['controller'])
        self.reader_refit_source_ids = reader_refit_sources(self.settings)
        if self.stage not in ('train', 'evaluate', 'all'):
            raise BlockedError(f'Unknown stage: {self.stage}')
        if self.settings.get('run_kind') != 'full_training':
            raise BlockedError('This runner requires full_training settings; use ega_check.py for bounded checks')
        for section, field in (('reader', 'max_windows'), ('controller', 'max_teacher_states'),
                               ('controller', 'max_teacher_pairs')):
            if self.config[section].get(field) is not None:
                raise BlockedError(f'Full training cannot use {section}.{field} slicing')
        if not self.config.get('discriminator', {}).get('enabled'):
            raise BlockedError('Full EGA requires discriminator token GRPO')
        if self.config['discriminator'].get('max_steps') is not None:
            raise BlockedError('Full discriminator training cannot use a bounded step cap')
        if self.config['controller'].get('grpo', {}).get('enabled'):
            raise BlockedError('The EGA controller uses paired regression; GRPO belongs to the discriminator')
        if self.config['controller']['grpo'].get('max_steps') is not None:
            raise BlockedError('Full training cannot use a bounded GRPO step cap')
        if self.evaluation.get('protocol') not in ('native', 'transfer', 'nested'):
            raise BlockedError('Declare native, transfer, or nested evaluation')
        if self.task_id == 'eatd' and (self.source_id != 'daic'
                                     or self.evaluation['protocol'] != 'transfer'
                                     or self.settings['reader_sources'] not in (
                                         ['daic'], ['daic', 'mmpsy_depression'])):
            raise BlockedError('EATD uses DAIC task training and declared source-only Reader supervision, followed by target testing')
        if self.evaluation['protocol'] == 'nested' and (
                self.evaluation.get('outer_folds') != 5 or self.evaluation.get('inner_folds') != 4):
            raise BlockedError('The declared participant protocol requires outer5/inner4')
        if any(self.settings.get(key) for key in ('frontend_checkpoint', 'teacher_checkpoint', 'fitted_checkpoint')) and self.evaluation['protocol'] == 'nested':
            raise BlockedError('Frontend checkpoint reuse requires unchanged source train/dev partitions, not new folds')
        if self.source_id not in self.settings['reader_sources']:
            raise BlockedError('reader_sources must include the controller training task')
        from tmm.support.reader_state import reader_state_mode
        soft_reader_state = reader_state_mode(self.config['controller']) == 'structured_soft'
        if self.settings.get('plain_teacher_checkpoint') and (
                (self.settings.get('reader_refit') is not True and not soft_reader_state)
                or not self.settings.get('frontend_checkpoint')
                or self.settings.get('teacher_checkpoint') or self.settings.get('fitted_checkpoint')
                or self.evaluation['protocol'] != 'transfer' or self.source_id != 'daic'
                or self.task_id != 'eatd' or self.source_task['track'] != 'A+T'):
            raise BlockedError('Plain Teacher reuse requires DAIC-to-EATD Reader refitting or frozen soft-state re-encoding')

    def prepare(self):
        from tmm.support.data import digest, load_records, read_jsonl, participant_folds
        self.records = load_records(ROOT, self.task)
        self.source = (self.records if self.source_id == self.task_id
                       else load_records(ROOT, self.source_task))
        if any(r.get('smoke_only') for r in self.records + self.source):
            raise BlockedError('Full training rejects temporary smoke-only participant partitions')
        if self.task_id.startswith('mmpsy_'):
            declaration = self.settings.get('dataset_partition', {})
            frozen = load_json(ROOT / declaration.get('manifest', 'manifests/sources/mmpsy_joint811_seed27.json'))
            if declaration.get('protocol_id') != frozen['protocol_id'] or declaration.get('seed') != 27 or frozen['seed'] != 27:
                raise BlockedError('MMPsy must retain the frozen public-subset partition at seed27')
            assignment = {r['participant']: r['split'] for r in frozen['participants']}
            if {r['participant'] for r in self.records} != set(assignment):
                raise BlockedError('MMPsy does not contain every frozen public-subset participant')
            if any(r['split'] != assignment.get(r['participant']) for r in self.records):
                raise BlockedError('MMPsy record differs from the frozen participant partition')
        target_people = people(self.records)
        if self.source_id != self.task_id and people(self.source) & target_people:
            raise BlockedError('Transfer source and target participant groups overlap')
        self.annotations = read_jsonl(self.annotations_path)
        self.supplements = []
        for source_id in self.settings['reader_sources']:
            if source_id == self.source_id:
                continue
            source_task = self.config['tasks'][source_id]
            extra = load_records(ROOT, source_task)
            if self.task_id == 'eatd' and source_id == 'mmpsy_depression':
                from tmm.support.reader_sources import mmpsy_reader_records
                partition = load_json(ROOT / 'manifests/sources/mmpsy_joint811_seed27.json')
                extra = mmpsy_reader_records(extra, partition, target_people)
            self.supplements.extend(r for r in extra if r['split'] in ('train', 'dev')
                                    and (r['family'], r['participant']) not in target_people)
        # Reader source restoration is scoped to checkpoint declarations; it
        # does not permit a new source list in ordinary non-refit settings.
        self.checkpoint_declarations = {}
        for setting in ('frontend_checkpoint', 'teacher_checkpoint', 'fitted_checkpoint', 'plain_teacher_checkpoint'):
            if not self.settings.get(setting):
                continue
            path = (ROOT / self.settings[setting]).resolve()
            declaration_path = path.parent/'settings.json'
            if not path.is_file() or not declaration_path.is_file():
                raise BlockedError(f'{setting} requires final.pt and its original settings.json')
            previous = load_json(declaration_path)
            self.checkpoint_declarations[setting] = (path, previous)
        same_source_reader_refit = (
            self.settings.get('reader_refit') is True
            and self.settings.get('reader_refit_sources') is None
            and self.settings.get('frontend_checkpoint')
            and not self.settings.get('teacher_checkpoint') and not self.settings.get('fitted_checkpoint'))
        restoring_for_optimizer_refit = (
            same_source_reader_refit
            and self.settings.get('reader', {}).get('training_mode') == 'participant_minibatch_v1')
        restoring_for_joint_refit = (
            same_source_reader_refit
            and not restoring_for_optimizer_refit
            and self.settings.get('reader', {}).get('boundary_likelihood') == 'global_legal_joint_v1'
            and self.settings.get('reader', {}).get('speaker_normalization') == 'ontology_casefold_v1')
        restoring_for_span_refit = (
            same_source_reader_refit and not restoring_for_joint_refit and not restoring_for_optimizer_refit
            and self.settings.get('reader', {}).get('relation_conditioning') == 'predicted_span_mean')
        restoring_for_same_sources = (restoring_for_span_refit or restoring_for_joint_refit
                                      or restoring_for_optimizer_refit)
        if self.settings.get('reader_refit', False) is False or restoring_for_same_sources:
            from tmm.support.reader_sources import restored_reader_sources
            # Refit the same verified source roster; this does not authorize a
            # new source extension or alter the actual Reader-refit request.
            source_settings = (dict(self.settings, reader_refit=False)
                               if restoring_for_same_sources else self.settings)
            # A plain Teacher supplies native scores, not the current Reader.
            # Its original Reader source declaration may therefore differ.
            declarations = [restored_reader_sources(source_settings, previous)
                            for setting, (_, previous) in self.checkpoint_declarations.items()
                            if setting != 'plain_teacher_checkpoint']
            if any(d is not None for d in declarations):
                if any(d != declarations[0] for d in declarations):
                    raise BlockedError('Selected checkpoints disagree on their declared Reader sources')
                self.restored_reader_source_declaration = declarations[0]
                self.reader_refit_source_ids = list(declarations[0]['new_sources'])
        self.refit_supplements = []
        self.refit_source_provenance = None
        if self.reader_refit_source_ids is not None:
            from tmm.support.reader_sources import mmpsy_reader_records
            supplement_task = self.config['tasks']['mmpsy_depression']
            manifest = ROOT / supplement_task['manifest']
            partition_path = ROOT / 'manifests/sources/mmpsy_joint811_seed27.json'
            self.refit_supplements = mmpsy_reader_records(
                load_records(ROOT, supplement_task), load_json(partition_path), target_people)
            self.refit_source_provenance = {
                'task': 'mmpsy_depression', 'manifest': str(manifest),
                'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                'partition_manifest': str(partition_path),
                'partition_sha256': hashlib.sha256(partition_path.read_bytes()).hexdigest(),
                'records_sha256': digest([{k: v for k, v in r.items() if k != 'index_seconds'}
                                          for r in self.refit_supplements]),
                'partitions': ['train', 'dev'], 'modalities': ['T']}
            if (self.restored_reader_source_declaration is not None
                    and self.refit_source_provenance !=
                    self.restored_reader_source_declaration['supplement_provenance']):
                raise BlockedError('Restored Reader supplement content or partition identity changed')
        self.train = [r for r in self.source if r['split'] == 'train']
        self.dev = [r for r in self.source if r['split'] == 'dev']
        self.test = [r for r in self.records if r['split'] == 'test']
        if self.evaluation['protocol'] == 'nested':
            self.outer = list(participant_folds(self.records, 5, self.seed))
            self.fold_counts = []
            for number, (trainval, test) in enumerate(self.outer, 1):
                inner = list(participant_folds(trainval, 4, self.seed + number))
                counts = []
                for train, dev in inner:
                    counts.append(self.check_reader(repartition(train, 'train'), repartition(dev, 'dev'), fold=True))
                self.fold_counts.append({'outer': number, 'train_participants': len(people(trainval)),
                                         'test_participants': len(people(test)), 'inner': counts})
        else:
            if not self.train or not self.dev:
                self.check_training_blockers.append('Separate source train and development participants are required')
            elif people(self.train) & people(self.dev):
                self.check_training_blockers.append('Source training and development participants overlap')
            else:
                self.reader_counts = self.check_reader(self.train, self.dev)
            if not self.test:
                self.check_evaluation_blockers.append('Native test labels are missing; development data cannot replace test evaluation')
            elif self.task_id == 'eatd' and len(people(self.test)) != 79:
                self.check_evaluation_blockers.append('EATD transfer evaluation must contain all 79 declared target participants')
        if not Path(self.model['path']).is_dir():
            self.check_training_blockers.append(f'Frozen pretrained checkpoint is unavailable: {self.model["path"]}')
            self.check_evaluation_blockers.append(f'Frozen pretrained checkpoint is unavailable: {self.model["path"]}')
        fingerprints = {}
        paths = [self.annotations_path, ROOT / self.settings.get('config', 'configs/experiments.json')]
        for spec in (self.source_task, self.task):
            if spec.get('manifest') and (ROOT / spec['manifest']).is_file():
                paths.append(ROOT / spec['manifest'])
            if spec.get('format') == 'daic':
                for partition in spec['splits'].values():
                    paths.extend((ROOT / partition['ids_file'], Path(partition['label_csv'])))
        paths.extend(path for path in Path(self.model['path']).glob('*.json')
                     if path.name in ('config.json', 'generation_config.json', 'model.safetensors.index.json',
                                      'pytorch_model.bin.index.json', 'tokenizer_config.json'))
        for path in paths:
            fingerprints[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        normalized = lambda rows: [{k: v for k, v in r.items() if k != 'index_seconds'} for r in rows]
        stable_settings = {k: v for k, v in self.settings.items() if k not in ('stage', 'gpu')}
        code = {p.relative_to(ROOT / 'code/tmm').as_posix(): p.read_text(encoding='utf-8')
                for p in (ROOT / 'code/tmm').rglob('*.py')}
        code['scripts/ega_run.py'] = Path(__file__).read_text(encoding='utf-8')
        from tmm.support.experiments import input_provenance
        self.provenance = input_provenance(self.source + self.records + self.supplements, self.model)
        for setting in ('frontend_checkpoint', 'teacher_checkpoint', 'fitted_checkpoint', 'plain_teacher_checkpoint'):
            if not self.settings.get(setting):
                continue
            path, previous = self.checkpoint_declarations[setting]
            old_settings = previous.get('settings', {})
            if (setting == 'frontend_checkpoint' and restoring_for_same_sources
                    and self.restored_reader_source_declaration is not None):
                old_reader = copy.deepcopy(old_settings.get('reader', {}))
                if restoring_for_optimizer_refit:
                    from tmm.support.experiments import validate_reader_optimizer_refit
                    validate_reader_optimizer_refit(old_reader, self.settings.get('reader', {}))
                elif restoring_for_joint_refit:
                    from tmm.support.experiments import validate_joint_reader_refit
                    validate_joint_reader_refit(old_reader, self.settings.get('reader', {}))
                else:
                    if old_reader.get('relation_conditioning', 'query_only') != 'query_only':
                        raise BlockedError('Span Reader refitting requires the original query-only Reader')
                    old_reader['relation_conditioning'] = 'predicted_span_mean'
                    if old_reader != self.settings.get('reader', {}):
                        raise BlockedError('Span Reader refitting may change only relation conditioning on its verified sources')
            if self.settings.get('reader_refit_sources') is not None and (
                    old_settings.get('reader_refit_sources') is not None
                    or previous.get('reader_source_declaration') is not None):
                raise BlockedError('Reader source extension requires its original DAIC-only frontend declaration')
            if (previous.get('schema') != 'ega-full-run-v1'
                    or previous.get('source_task') != self.source_task
                    or previous.get('model') != self.model
                    or old_settings.get('training_task', old_settings.get('task')) != self.source_id
                    or old_settings.get('seed') != self.seed
                    or old_settings.get('reader_sources') != self.settings['reader_sources']
                    or old_settings.get('protocol', {}) != self.settings.get('protocol', {})
                    or (self.source_task['track'] == 'F' and
                        old_settings.get('adapter', {}) != self.settings.get('adapter', {}))
                    or (setting == 'teacher_checkpoint' and
                        gate_calibration(old_settings) != gate_calibration(self.settings))):
                raise BlockedError(f'{setting} source/model/protocol declaration changed')
            # Reuse the input identity already computed for this run; no new media scan.
            if previous.get('input_provenance') != self.provenance:
                raise BlockedError(f'{setting} data/media/model identity changed')
            if setting == 'fitted_checkpoint' and fitting_settings(old_settings) != fitting_settings(self.settings):
                raise BlockedError('Fitted checkpoint reuse only permits changing the source cutoff objective')
            checksum = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024*1024), b''):
                    checksum.update(block)
            origin = {'checkpoint': str(path), 'checkpoint_sha256': checksum.hexdigest(),
                      'run_fingerprint': previous['fingerprint'],
                      'source_identity': digest((self.source_task, self.model,
                                                old_settings.get('protocol', {}), self.provenance))}
            if self.restored_reader_source_declaration is not None:
                origin['reader_source_declaration'] = self.restored_reader_source_declaration
            if setting == 'frontend_checkpoint':
                self.frontend_origin = origin
                if self.settings.get('reader_refit_sources') is not None:
                    # The ordinary reader_sources check above remains exact. In
                    # this scoped extension it is the original ['daic'] roster;
                    # native source records were loaded once and are reused.
                    self.reader_refit_origin = {
                        'schema': 'ega-reader-source-refit-v1',
                        'original_sources': list(old_settings['reader_sources']),
                        'new_sources': self.reader_refit_source_ids,
                        'original_reader_train': self.original_readers(self.train, self.dev)[0],
                        'original_reader_dev': self.original_readers(self.train, self.dev)[1],
                        'original_input_provenance': self.provenance,
                        'supplement_provenance': self.refit_source_provenance,
                        'frontend_checkpoint_sha256': origin['checkpoint_sha256']}
            elif setting == 'teacher_checkpoint':
                self.teacher_origin, self.teacher_declaration = origin, previous
            elif setting == 'fitted_checkpoint':
                protocol_path = path.parent/'fit_protocol.json'
                protocol_bytes = protocol_path.read_bytes()
                protocol = json.loads(protocol_bytes)
                if protocol.get('schema') != 'ega-fit-v1' or not protocol.get('signature'):
                    raise BlockedError('Fitted checkpoint requires its completed fitting protocol')
                self.fitted_origin = dict(origin, fit_signature=protocol['signature'],
                                          fit_protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest())
            else:
                status = load_json(path.parent/'status.json')
                protocol_bytes = (path.parent/'fit_protocol.json').read_bytes()
                protocol = json.loads(protocol_bytes)
                old_teacher = copy.deepcopy(protocol.get('teacher_config'))
                raw = protocol.get('score_calibration_provenance') or {}
                components = protocol.get('frontend_components') or {}
                if (path.name != 'final.pt' or status.get('status') != 'completed'
                        or status.get('fingerprint') != previous['fingerprint']
                        or protocol.get('schema') != 'ega-fit-v1' or not protocol.get('signature')
                        or not isinstance(old_teacher, dict)
                        or old_teacher.get('teacher_rollout') != {'mode': 'seed_aligned_audio'}
                        or not components.get('reader') or not components.get('discriminator')
                        or not raw.get('raw_files') or not raw.get('source_records_signature')
                        or old_settings.get('discriminator') != self.settings.get('discriminator')):
                    raise BlockedError('Plain Teacher reuse requires its completed native raw rollout and frozen discriminator identity')
                old_teacher['score_calibration'] = {'mode': 'disabled'}
                self.plain_teacher_origin = dict(origin,
                    schema='ega-plain-teacher-origin-v1', fit_signature=protocol['signature'],
                    fit_protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
                    source_reader_fingerprint=components['reader'],
                    source_reader_gate=old_teacher.get('reader_gate'), source_teacher_config=old_teacher,
                    raw_files=raw['raw_files'], source_records_signature=raw['source_records_signature'],
                    native_identity={'backbone_spec': self.model, 'protocol': self.config['protocol'],
                        'instruction': self.source_task['instruction'],
                        'discriminator_fingerprint': components['discriminator'],
                        'track': self.source_task['track'], 'renderers': [0, 1, 2]})
        self.fingerprint = digest({'settings': stable_settings, 'source': normalized(self.source),
                                   'target': normalized(self.records), 'reader_supplements': normalized(self.supplements),
                                   **({'reader_refit_supplements': normalized(self.refit_supplements),
                                       'reader_refit_source_provenance': self.refit_source_provenance}
                                      if self.reader_refit_source_ids is not None else {}),
                                   'files': fingerprints, 'input_provenance': self.provenance, 'code': code,
                                   **({'frontend_origin': self.frontend_origin} if self.frontend_origin else {}),
                                   **({'teacher_origin': self.teacher_origin} if self.teacher_origin else {}),
                                   **({'plain_teacher_origin': self.plain_teacher_origin} if self.plain_teacher_origin else {}),
                                   **({'fitted_origin': self.fitted_origin} if self.fitted_origin else {})})
        self.out = (Path(self.args.output).resolve() if self.args.output
                    else ROOT / 'runs/ega' / self.task_id / self.model_id / str(self.seed) / self.fingerprint)
        if self.stage == 'evaluate':
            checkpoint_dirs = ([self.out / f'outer_{i}' for i in range(1, 6)]
                               if self.evaluation['protocol'] == 'nested' else [self.out])
            self.check_evaluation_blockers.extend(f'Missing trained checkpoint: {path / "final.pt"}'
                                                 for path in checkpoint_dirs if not (path / 'final.pt').is_file())

    def original_readers(self, train, dev):
        reader_train = list(train) + [r for r in self.supplements if r['split'] == 'train']
        reader_dev = list(dev) + ([r for r in self.supplements if r['split'] == 'dev'] if dev else [])
        if people(reader_train) & people(reader_dev):
            raise BlockedError('Reader training and development participants overlap')
        return reader_train, reader_dev

    def readers(self, train, dev):
        reader_train, reader_dev = self.original_readers(train, dev)
        reader_train += [r for r in self.refit_supplements if r['split'] == 'train']
        if dev:
            reader_dev += [r for r in self.refit_supplements if r['split'] == 'dev']
        if people(reader_train) & people(reader_dev):
            raise BlockedError('Expanded Reader training and development participants overlap')
        return reader_train, reader_dev

    def check_reader(self, train, dev, fold=False):
        from tmm.support.reader_inputs import annotations_for, validate_config
        from tmm.support.data import seed_views, ObservationError
        validate_config(self.config['reader'])
        counts = {'train_records': len(train), 'dev_records': len(dev)}
        for split, rows in (('train', train), ('dev', dev)):
            excluded = []
            for record in rows:
                try:
                    seed_views(record, self.source_task['track'])
                except ObservationError as exc:
                    excluded.append({'id': record['id'], 'reason': str(exc)})
            counts[split + '_eligible_seed_records'] = len(rows) - len(excluded)
            counts[split + '_input_exclusions'] = excluded
            if rows and len(rows) == len(excluded):
                raise BlockedError(f'No eligible controller seed observations for {split}')
        reader_train, reader_dev = self.readers(train, dev)
        annotations = split_annotations(self.annotations, reader_train + reader_dev) if fold else self.annotations
        for name, rows in (('reader_train', reader_train), ('reader_dev', reader_dev)):
            windows = annotations_for(rows, annotations, self.config['reader'])
            if rows and not windows:
                raise BlockedError(f'No reviewed reader supervision is available for {name}')
            counts[name + '_records'] = len(rows)
            counts[name + '_windows'] = len(windows)
        return counts

    def declaration(self):
        from tmm.support.reader_sources import refit_declaration
        return {'schema': 'ega-full-run-v1', 'fingerprint': self.fingerprint, 'settings': self.settings,
                'task': self.task, 'source_task': self.source_task, 'model': self.model,
                'input_provenance': self.provenance,
                'frontend_origin': self.frontend_origin,
                'teacher_origin': self.teacher_origin,
                'plain_teacher_origin': self.plain_teacher_origin,
                'fitted_origin': self.fitted_origin,
                'reader_refit_origin': refit_declaration(self.reader_refit_origin),
                'reader_source_declaration': (self.restored_reader_source_declaration
                                              or refit_declaration(self.reader_refit_origin)),
                'method': 'Supervised Reader; Screening discriminator token GRPO with LoRA; freeze both; paired-regression Controller.'}

    def bind_output(self):
        from tmm.support.data import write_json
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / 'settings.json'
        if path.is_file() and load_json(path).get('fingerprint') != self.fingerprint:
            raise BlockedError('Output contains a different data/config/code fingerprint; select another --output')
        if not path.is_file() and any(self.out.glob('*.pt')):
            raise BlockedError('Output contains unbound checkpoints; select another --output')
        write_json(path, self.declaration())
        self.output_bound = True

    def load_backbone(self):
        if self.backbone is None:
            import numpy as np
            import torch
            from tmm.support.models import Backbone
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.set_num_threads(self.settings['cpu_threads'])
            self.backbone = Backbone(self.model, self.config['protocol'], self.source_task['instruction'])
        self.backbone.instruction = self.source_task['instruction']
        return self.backbone

    def fit(self, train, dev, out, selected=None, fold=False):
        import torch
        from tmm.support.experiments import (fit_bundle, frozen_component_identity, load_teacher_cache,
                                     bundle_budget_policy, bundle_target_policy)
        torch.manual_seed(self.seed)
        reader_train, reader_dev = self.readers(train, dev)
        config = copy.deepcopy(self.config)
        if fold:
            path = out / 'reader/items.jsonl'
            write_jsonl(path, split_annotations(self.annotations, reader_train + reader_dev))
            config['reader']['annotations'] = str(path)
        if self.fitted_origin:
            previous = torch.load(self.fitted_origin['checkpoint'], map_location='cpu', weights_only=False)
            self.validate_restored_reader_source(previous, self.fitted_origin)
            if previous.get('fit_signature') != self.fitted_origin['fit_signature']:
                raise BlockedError('Fitted checkpoint does not match its original fitting protocol')
            if previous.get('fitting', {}).get('algorithm') != 'paired_supervision':
                raise BlockedError('Fitted checkpoint does not contain the completed paired Controller')
            frozen_component_identity(previous, self.model, train, dev, reader_train, reader_dev,
                                      self.source_task['track'], config, self.seed)
            controller, bundle, _ = fit_bundle(self.load_backbone(), train, dev, self.source_task['track'],
                                               config, seed=self.seed, reader_train=reader_train,
                                               reader_dev=reader_dev, checkpoint_dir=out, resume=self.args.resume,
                                               trained_bundle=previous, fitted_origin=self.fitted_origin)
            bundle_budget_policy(bundle, self.config['controller'])
            bundle_target_policy(bundle, self.config['controller'])
            return controller, bundle
        component_origin = self.frontend_origin or self.teacher_origin
        if component_origin and self.frozen_components is None:
            previous = torch.load(component_origin['checkpoint'], map_location='cpu', weights_only=False)
            self.validate_restored_reader_source(previous, component_origin)
            # Do not retain or restore the old Controller, policy, cutoff or predictions.
            keys = ('schema', 'reader', 'discriminator', 'adapter', 'train_participants', 'dev_participants')
            self.frozen_components = {key: previous[key] for key in keys}
            if previous.get('reader_refit_provenance'):
                self.frozen_components['reader_refit_provenance'] = previous['reader_refit_provenance']
            del previous
        if self.teacher_origin and self.teacher_cache is None:
            previous = torch.load(self.teacher_origin['checkpoint'], map_location='cpu', weights_only=False)
            self.validate_restored_reader_source(previous, self.teacher_origin)
            ids = frozen_component_identity(previous, self.model, train, dev, reader_train, reader_dev,
                                            self.source_task['track'], config, self.seed)
            frontend_ids = frozen_component_identity(self.frozen_components, self.model, train, dev,
                                                      reader_train, reader_dev, self.source_task['track'], config, self.seed)
            if ids != frontend_ids:
                raise BlockedError('Teacher checkpoint belongs to another frozen frontend')
            old_settings = self.teacher_declaration['settings']
            old_controller = copy.deepcopy(load_json(ROOT / old_settings.get('config', 'configs/experiments.json'))['controller'])
            merge_config(old_controller, old_settings['controller'])
            old_controller['seed'] = old_settings['seed']
            self.teacher_cache, self.validated_teacher_origin = load_teacher_cache(
                self.teacher_origin['checkpoint'], previous, train, dev, self.source_task['track'],
                config, old_controller, self.teacher_origin, ids)
            del previous
        if self.plain_teacher_origin and not self.plain_teacher_verified:
            from tmm.support.experiments import _record_signature
            from tmm.support import grpo
            origin = self.plain_teacher_origin
            previous = torch.load(origin['checkpoint'], map_location='cpu', weights_only=False)
            if (previous.get('schema') != 'ega-v2' or previous.get('fit_signature') != origin['fit_signature']
                    or previous.get('reader', {}).get('fingerprint') != origin['source_reader_fingerprint']
                    or previous.get('discriminator', {}).get('fingerprint') != origin['native_identity']['discriminator_fingerprint']
                    or previous.get('discriminator', {}).get('backbone_spec') != self.model
                    or self.frozen_components['discriminator']['fingerprint'] != origin['native_identity']['discriminator_fingerprint']
                    or previous.get('source_records_signature') != origin['source_records_signature']
                    or origin['source_records_signature'] != grpo.signature((_record_signature(train), _record_signature(dev)))
                    or previous.get('score_calibration_provenance', {}).get('raw_files') != origin['raw_files']
                    or previous.get('policy', {}).get('reader_gate') != origin['source_reader_gate']):
                raise BlockedError('Plain Teacher final checkpoint disagrees with its native source, raw files or fitting protocol')
            del previous
            self.plain_teacher_verified = True
        controller, bundle, _ = fit_bundle(self.load_backbone(), train, dev, self.source_task['track'],
                                           config, seed=self.seed, selected=selected,
                                           reader_train=reader_train, reader_dev=reader_dev,
                                           checkpoint_dir=out, resume=self.args.resume,
                                           frozen_components=self.frozen_components,
                                           frontend_origin=component_origin,
                                           pair_cache=self.teacher_cache, teacher_origin=self.validated_teacher_origin,
                                           **({'plain_teacher_origin': self.plain_teacher_origin}
                                              if self.plain_teacher_origin is not None else {}),
                                           **({'reader_refit_origin': self.reader_refit_origin}
                                              if self.reader_refit_origin is not None else {}))
        bundle_budget_policy(bundle, self.config['controller'])
        bundle_target_policy(bundle, self.config['controller'])
        if bundle.get('schema') != 'ega-v2' or not bundle.get('discriminator'):
            raise RuntimeError('Completed EGA checkpoint has no trained discriminator')
        if bundle['fitting'].get('algorithm') != 'paired_supervision':
            raise RuntimeError('The EGA controller must use its paired regression objective')
        return controller, bundle

    def validate_restored_reader_source(self, bundle, origin):
        if self.restored_reader_source_declaration is not None:
            from tmm.support.reader_sources import validate_restored_reader_bundle
            validate_restored_reader_bundle(bundle, self.restored_reader_source_declaration,
                                             origin['checkpoint_sha256'])

    def restore(self, out):
        import torch
        from tmm.support.experiments import restore_bundle, bundle_budget_policy, bundle_target_policy
        path = out / 'final.pt'
        if not path.is_file():
            raise BlockedError(f'No trained EGA checkpoint: {path}; run --stage train first')
        bundle = torch.load(path, map_location='cpu', weights_only=False)
        if bundle.get('schema') != 'ega-v2' or not bundle.get('discriminator'):
            raise BlockedError('Final checkpoint lacks the trained Screening discriminator')
        if bundle['fitting'].get('algorithm') != 'paired_supervision':
            raise BlockedError('Final checkpoint does not use the paired-regression EGA controller')
        declaration = out / 'fit_protocol.json'
        protocol = load_json(declaration) if declaration.is_file() else {}
        if not declaration.is_file() or bundle.get('fit_signature') != protocol.get('signature'):
            raise BlockedError('Final checkpoint does not match its saved fitting protocol')
        from tmm.support.reader_state import reader_state_mode
        if reader_state_mode(bundle['policy']) != reader_state_mode(self.config['controller']):
            raise BlockedError('Final Controller Reader encoding does not match the requested runtime payload mode')
        budget_policy = bundle_budget_policy(bundle, self.config['controller'])
        if budget_policy and protocol.get('budget_profile') != budget_policy['budget_profile']:
            raise BlockedError('Final Controller budget profile differs from its resolved fitting protocol')
        target_policy = bundle_target_policy(bundle, self.config['controller'])
        if target_policy and protocol.get('target_profile') != target_policy['target_profile']:
            raise BlockedError('Final Controller target scale differs from its resolved fitting protocol')
        controller = restore_bundle(self.load_backbone(), bundle)
        return controller, bundle

    def export_cases(self, predictions, records, bundle, out, outer_fold=None):
        from tmm.support.case_records import export_case_records
        metadata = {'run_fingerprint': self.fingerprint, 'task': self.task_id, 'model': self.model_id,
                    'source_task': self.source_id, 'track': self.task['track'],
                    'evaluation': self.evaluation, 'outer_fold': outer_fold,
                    'cohort': {'records': len(records), 'participants': len(people(records))},
                    'checkpoint': str(out / 'final.pt'), 'fit_signature': bundle.get('fit_signature'),
                    'frontend_components': bundle.get('frontend_components'), 'policy': bundle['policy']}
        try:
            return export_case_records(predictions, records, out, metadata, ROOT / 'cases/index')
        except (OSError, ValueError) as exc:
            # Predictions are already durable. Logging never changes a model or
            # forces another score; a subsequent resume can repair missing cases.
            print(json.dumps({'case_export_warning': str(exc), 'output': str(out)}), flush=True)

    def evaluate(self, records, controller, bundle, out, outer_fold=None):
        from tmm.support.data import write_json, ObservationError
        from tmm.controller import acquire
        from tmm.support.experiments import row, bundle_budget_policy, bundle_target_policy
        from tmm.support.reader_training import index_reader_positions
        bundle_budget_policy(bundle, self.config['controller'])
        bundle_target_policy(bundle, self.config['controller'])
        out.mkdir(parents=True, exist_ok=True)
        path = out / 'predictions.json'
        saved = load_json(path) if path.is_file() else {'schema': 'ega-predictions-v1',
                'fingerprint': self.fingerprint, 'threshold': bundle['threshold'], 'seed': [], 'ega': []}
        if saved.get('fingerprint') != self.fingerprint or saved.get('threshold') != bundle['threshold']:
            raise BlockedError('Cached predictions do not match this run or classifier threshold')
        expected = {r['id'] for r in records}
        done = {r['id'] for r in saved['seed']}
        if (len(done) != len(saved['seed']) or done != {r['id'] for r in saved['ega']}
                or len(saved['ega']) != len(done) or not done <= expected):
            raise BlockedError('Cached predictions do not match the declared evaluation cohort')
        self.export_cases(saved, records, bundle, out, outer_fold)
        backbone = self.load_backbone()
        backbone.instruction = self.task['instruction']
        index_reader_positions(records, bundle['reader']['config'])
        for record in records:
            if record['id'] in done:
                continue
            try:
                pre, post = acquire(backbone, record, self.task['track'], controller, bundle['policy'],
                                    self.settings['budget'], self.settings['acquisition_limit'])
            except ObservationError as exc:
                pre = post = {'p': .5, 'status': 'input_failure', 'probability_valid': False,
                              'error': str(exc), 'seconds': 0., 'gpu_seconds': 0.}
            saved['seed'].append(row(record, pre, bundle['threshold']))
            saved['ega'].append(row(record, post, bundle['threshold']))
            write_json(path, saved)
            self.export_cases({'fingerprint': self.fingerprint, 'threshold': bundle['threshold'],
                               'seed': saved['seed'][-1:], 'ega': saved['ega'][-1:]},
                              records, bundle, out, outer_fold)
            print(json.dumps({'stage': 'evaluate', 'record': record['id'], 'done': len(saved['seed']),
                              'total': len(records), 'status': post['status']}), flush=True)
        return saved

    def metrics(self, predictions):
        from tmm.support.metrics import classification, paired
        before, after = predictions['seed'], predictions['ega']
        cut = predictions.get('threshold', .5)
        return {'schema': 'ega-full-metrics-v1', 'fingerprint': self.fingerprint,
                'evaluation': self.evaluation, 'threshold': predictions.get('threshold'),
                'seed': classification(before, cut), 'ega': classification(after, cut),
                'transitions': paired(before, after, cut),
                'mean_seed_seconds': statistics.mean(r.get('seconds', 0.) for r in before),
                'mean_ega_seconds': statistics.mean(r.get('seconds', 0.) for r in after),
                'mean_acquisition_steps': statistics.mean(r.get('steps', 0) for r in after)}

    def native(self):
        from tmm.support.data import write_json
        if self.stage in ('train', 'all'):
            if self.check_training_blockers:
                raise BlockedError('; '.join(self.check_training_blockers))
            controller, bundle = self.fit(self.train, self.dev, self.out)
            calibration_rows(bundle, self.dev)
            write_json(self.out / 'status.json', {'status': 'trained', 'fingerprint': self.fingerprint,
                       'checkpoint': str(self.out / 'final.pt'), 'teacher_pair_count': bundle['teacher_pair_count']})
        else:
            controller, bundle = self.restore(self.out)
        if self.stage in ('evaluate', 'all'):
            if self.check_evaluation_blockers:
                raise BlockedError('; '.join(self.check_evaluation_blockers))
            predictions = self.evaluate(self.test, controller, bundle, self.out)
            metrics = self.metrics(predictions)
            write_json(self.out / 'metrics.json', metrics)
            return metrics
        return {'status': 'trained', 'checkpoint': str(self.out / 'final.pt')}

    def nested(self):
        from tmm.support.data import participant_folds, write_json
        combined = {'schema': 'ega-nested-predictions-v1', 'fingerprint': self.fingerprint,
                    'seed': [], 'ega': []}
        for number, (trainval, test) in enumerate(self.outer, 1):
            outer_out = self.out / f'outer_{number}'
            inner_folds = list(participant_folds(trainval, 4, self.seed + number))
            plan = {'outer': number, 'train': sorted(people(trainval)), 'test': sorted(people(test)),
                    'inner': [{'train': sorted(people(t)), 'dev': sorted(people(d))} for t, d in inner_folds]}
            plan_path = outer_out / 'partitions.json'
            if plan_path.is_file() and load_json(plan_path) != json.loads(json.dumps(plan)):
                raise BlockedError('Saved nested participant assignments differ from the declared folds')
            write_json(plan_path, plan)
            selection_path = outer_out / 'selection.json'
            if self.stage in ('train', 'all'):
                if self.check_training_blockers:
                    raise BlockedError('; '.join(self.check_training_blockers))
                if selection_path.is_file():
                    selected = load_json(selection_path)['selected']
                else:
                    bundles, pooled, calibration_coverage = [], [], set()
                    for inner_number, (train, dev) in enumerate(inner_folds, 1):
                        inner_train, inner_dev = repartition(train, 'train'), repartition(dev, 'dev')
                        _, bundle = self.fit(inner_train, inner_dev, outer_out / f'inner_{inner_number}', fold=True)
                        pooled.extend(calibration_rows(bundle, inner_dev))
                        calibration_coverage.update(r['id'] for r in bundle['dev_seed_predictions'])
                        bundles.append(bundle)
                    if calibration_coverage != {r['id'] for r in trainval}:
                        raise BlockedError('Inner out-of-fold calibration does not cover every outer-training record')
                    selected = select_outer(bundles, pooled, self.config)
                    write_json(selection_path, {'selected': selected, 'calibration': pooled,
                               'calibration_total_records': len(trainval),
                               'calibration_valid_records': len(pooled),
                               'calibration_failed_records': len(trainval) - len(pooled),
                               'selection': 'Inner mean checkpoint epochs; median reader presence; pooled unaugmented seed cutoff.'})
                    del bundles, pooled
                controller, bundle = self.fit(repartition(trainval, 'train'), [], outer_out,
                                               selected=selected, fold=True)
            else:
                if not selection_path.is_file():
                    raise BlockedError(f'Missing inner-fold selection: {selection_path}')
                controller, bundle = self.restore(outer_out)
            if self.stage in ('evaluate', 'all'):
                predictions = self.evaluate(repartition(test, 'test'), controller, bundle, outer_out,
                                            outer_fold=number)
                for mode in ('seed', 'ega'):
                    combined[mode].extend(dict(r, outer_fold=number) for r in predictions[mode])
            print(json.dumps({'outer_fold': number, 'stage': self.stage, 'checkpoint': str(outer_out / 'final.pt')}), flush=True)
            del controller, bundle
            gc.collect()
        if self.stage == 'train':
            return {'status': 'trained', 'outer_folds': 5}
        if len(combined['seed']) != len(self.records):
            raise BlockedError('Outer predictions do not cover every target record')
        write_json(self.out / 'predictions.json', combined)
        metrics = self.metrics(combined)
        write_json(self.out / 'metrics.json', metrics)
        return metrics

    def metadata(self):
        required_train = self.stage in ('train', 'all')
        required_eval = self.stage in ('evaluate', 'all')
        report = {'status': 'ready', 'check_only': True, 'task': self.task_id, 'model': self.model_id,
                  'stage': self.stage, 'fingerprint': self.fingerprint, 'output': str(self.out),
                  'training_ready': not self.check_training_blockers,
                  'evaluation_ready': not self.check_evaluation_blockers,
                  'training_blockers': self.check_training_blockers,
                  'evaluation_blockers': self.check_evaluation_blockers,
                  'source': self.source_id, 'reader_sources': self.settings['reader_sources'],
                  'frontend_origin': self.frontend_origin,
                  'teacher_origin': self.teacher_origin,
                  'plain_teacher_origin': self.plain_teacher_origin,
                  'fitted_origin': self.fitted_origin,
                  'train_records': len(self.train), 'dev_records': len(self.dev), 'test_records': len(self.test)}
        if self.evaluation['protocol'] == 'nested':
            report['participant_folds'] = self.fold_counts
        else:
            report['reader'] = getattr(self, 'reader_counts', {})
        if (required_train and not report['training_ready']) or (required_eval and not report['evaluation_ready']):
            report['status'] = 'blocked'
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True, help='Explicit JSON settings file; - reads stdin')
    parser.add_argument('--check', action='store_true', help='Validate metadata only; never loads the backbone')
    parser.add_argument('--show-settings', action='store_true', help='Print settings without loading data or models')
    parser.add_argument('--stage', choices=('train', 'evaluate', 'all'))
    parser.add_argument('--resume', action='store_true', help='Resume interrupted stages at saved safe boundaries')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--seed', type=int, help='Training seed; the frozen MMPsy split remains seed27')
    parser.add_argument('--output', help='Run directory override; guarded by data/config/code fingerprint')
    args = parser.parse_args()
    run = None
    try:
        settings = json.load(sys.stdin) if args.settings == '-' else load_json(args.settings)
        if args.seed is not None:
            settings['seed'] = args.seed
        if args.gpu is not None:
            settings['gpu'] = args.gpu
        if args.show_settings:
            print(json.dumps(settings, ensure_ascii=False, indent=2), flush=True)
            return 0
        os.environ['CUDA_VISIBLE_DEVICES'] = str(settings['gpu'])
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        run = EgaRun(settings, args)
        run.prepare()
        if args.check:
            report = run.metadata()
            print(json.dumps(report, ensure_ascii=False), flush=True)
            return 0 if report['status'] == 'ready' else 2
        run.bind_output()
        result = run.nested() if run.evaluation['protocol'] == 'nested' else run.native()
        from tmm.support.data import write_json
        write_json(run.out / 'status.json', {'status': 'trained' if run.stage == 'train' else 'completed',
                   'fingerprint': run.fingerprint, 'stage': run.stage, 'result': result})
        print(json.dumps({'status': 'trained' if run.stage == 'train' else 'completed',
                          'task': run.task_id, 'model': run.model_id, 'output': str(run.out)}, ensure_ascii=False), flush=True)
        return 0
    except (BlockedError, FileNotFoundError, ValueError) as exc:
        report = {'status': 'blocked', 'check_only': args.check, 'reason': str(exc)}
        if run is not None:
            report.update(task=run.task_id, model=run.model_id)
            if run.output_bound and not args.check:
                from tmm.support.data import write_json
                if (run.out / 'status.json').is_file():
                    report['previous_stage'] = load_json(run.out / 'status.json')
                write_json(run.out / 'status.json', report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 2
    except Exception as exc:
        if run is not None and run.output_bound and not args.check:
            from tmm.support.data import write_json
            write_json(run.out / 'status.json', {'status': 'failed', 'fingerprint': run.fingerprint, 'reason': str(exc)})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
