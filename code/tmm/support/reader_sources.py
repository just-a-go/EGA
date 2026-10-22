"""Metadata-only declaration and partition guard for one Reader source extension."""
import copy
import hashlib
import json


def reader_refit_sources(settings):
    requested = settings.get('reader_refit_sources')
    if requested is None:
        return None
    if (requested != ['daic', 'mmpsy_depression'] or settings.get('reader_refit') is not True
            or settings.get('task') != 'eatd'
            or settings.get('training_task', settings.get('task')) != 'daic'
            or settings.get('evaluation', {}).get('protocol') != 'transfer'
            or settings.get('reader_sources') != ['daic']
            or not settings.get('frontend_checkpoint')
            or settings.get('teacher_checkpoint') or settings.get('fitted_checkpoint')):
        raise ValueError('Reader source extension requires declared DAIC-to-EATD Reader-only refitting')
    return list(requested)


def restored_reader_sources(settings, previous):
    """A declaration may request loading; the actual bundle must verify it later."""
    declaration = previous.get('reader_source_declaration') or previous.get('reader_refit_origin')
    if declaration is None:
        return None
    if (settings.get('reader_refit', False) is not False or settings.get('reader_refit_sources') is not None
            or settings.get('task') != 'eatd'
            or settings.get('training_task', settings.get('task')) != 'daic'
            or settings.get('evaluation', {}).get('protocol') != 'transfer'
            or settings.get('reader_sources') != ['daic']
            or declaration.get('schema') != 'ega-reader-source-refit-v1'
            or declaration.get('original_sources') != ['daic']
            or declaration.get('new_sources') != ['daic', 'mmpsy_depression']):
        raise ValueError('Chinese Reader restore requires its declared source-only checkpoint')
    supplement = declaration.get('supplement_provenance', {})
    if (supplement.get('task') != 'mmpsy_depression' or supplement.get('partitions') != ['train', 'dev']
            or supplement.get('modalities') != ['T']):
        raise ValueError('Restored Reader supplement source/partitions changed')
    return refit_declaration(declaration)


def validate_restored_reader_bundle(bundle, declaration, checkpoint_sha256):
    """Match declared source identity to the hashed checkpoint's real components."""
    provenance = bundle.get('reader_refit_provenance', {})
    fingerprint = hashlib.sha256(json.dumps({k: v for k, v in provenance.items()
                                             if k != 'fingerprint'}, sort_keys=True).encode()).hexdigest()
    if (not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64
            or provenance.get('source_declaration') != declaration
            or provenance.get('fingerprint') != fingerprint
            or provenance.get('reader') != bundle.get('reader', {}).get('fingerprint')
            or provenance.get('discriminator') != bundle.get('discriminator', {}).get('fingerprint')
            or provenance.get('original_reader') != bundle.get('discriminator', {}).get('reader_fingerprint')):
        raise ValueError('Chinese Reader source declaration does not match the hashed checkpoint provenance')


def mmpsy_reader_records(records, partition, target_people):
    """Retain only frozen source TRAIN/DEV; never silently repartition a user."""
    if partition.get('protocol_id') != 'mmpsy-public-joint811-seed27-v1' or partition.get('seed') != 27:
        raise ValueError('Reader supplement requires the frozen MMPsy participant partition')
    assignment = {row['participant']: row['split'] for row in partition['participants']}
    if len(assignment) != len(partition['participants']) or {r['participant'] for r in records} != set(assignment):
        raise ValueError('MMPsy Reader supplement participant inventory changed')
    if any(r['family'] != 'mmpsy' or r['split'] != assignment[r['participant']]
           or any(v['modality'] != 'T' for v in r['views']) for r in records):
        raise ValueError('MMPsy Reader supplement source/partition/text identity changed')
    selected = [r for r in records if r['split'] in ('train', 'dev')]
    if any((r['family'], r['participant']) in target_people for r in selected):
        raise ValueError('Reader supplement overlaps target participant identities')
    if ({(r['family'], r['participant']) for r in selected if r['split'] == 'train'}
            & {(r['family'], r['participant']) for r in selected if r['split'] == 'dev'}):
        raise ValueError('Reader supplement participants cross TRAIN/DEV')
    return selected


def refit_declaration(origin):
    """Persist source identity without duplicating the old record payloads."""
    if origin is None:
        return None
    return copy.deepcopy({k: v for k, v in origin.items()
                          if k not in ('original_reader_train', 'original_reader_dev')})
