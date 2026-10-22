"""One bounded EMER adaptation and honest Reader/Controller case search."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def evaluation_metrics(cases):
    """Aggregate independent option decisions, not fixed A--F emotion classes."""
    import numpy as np
    conditions = ('seed', 'reader_seed', 'full', 'without_reader', 'same_acquired_without_reader')
    result = {}
    for condition in conditions:
        truth, probabilities, sample_scores, exact = [], [], [], []
        for case in cases:
            prediction = {
                'seed': case['seed_prediction'], 'reader_seed': case['reader_seed_prediction'],
                'full': case['full']['final'], 'without_reader': case['without_reader']['final'],
                'same_acquired_without_reader': case['same_acquired_without_reader'],
            }[condition]
            y, p = np.asarray(case['reference']['labels']), np.asarray(prediction['p'], dtype=float)
            if (y.shape != (6,) or p.shape != (6,) or not np.isin(y, (0, 1)).all()
                    or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()):
                raise ValueError('EMER metrics require six finite independent option probabilities per sample')
            predicted = p >= .5
            denominator = int(predicted.sum()+y.sum())
            sample_scores.append(2*int((predicted & (y == 1)).sum())/denominator if denominator else 1.)
            exact.append(bool(np.array_equal(predicted, y)))
            truth.extend(y.tolist())
            probabilities.extend(p.tolist())
        if not truth:
            raise ValueError('EMER evaluation partition is empty')
        y, p = np.asarray(truth), np.asarray(probabilities)
        predicted = p >= .5
        tp = int(((y == 1) & predicted).sum())
        tn = int(((y == 0) & ~predicted).sum())
        fp = int(((y == 0) & predicted).sum())
        fn = int(((y == 1) & ~predicted).sum())
        positive_f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        negative_f1 = 2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
        denominator = float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
        result[condition] = {
            'sample_f1': float(np.mean(sample_scores)), 'exact_match': float(np.mean(exact)),
            'macro_f1': (positive_f1+negative_f1)/2, 'brier': float(np.mean((p-y)**2)),
            'mcc': (tp*tn-fp*fn)/denominator if denominator else 0.,
            'exact_matches': sum(exact), 'samples': len(cases), 'option_decisions': len(truth),
        }
    return {'threshold': .5, 'conditions': result,
            'basis': 'Six independent candidate-option decisions per sample; candidate emotions vary by sample. '
                     'sample_f1 and exact_match average over samples; macro_f1 is the 0/1 binary-class '
                     'macro average after flattening all sample-option decisions; MCC and Brier use the same flattened decisions.',
            'scope': 'Fixed qualitative EMER development/case-search partition, not an untouched benchmark test.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', required=True)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--stage', choices=['all','train','evaluate','reader','asr'], default='all')
    parser.add_argument('--show-settings', action='store_true', help='Print effective settings without loading models')
    parser.add_argument('--reuse-trained', type=Path,
                        help='Reuse an unchanged trained parent after an inference-only implementation repair')
    args = parser.parse_args()
    settings = json.load(sys.stdin) if args.settings == '-' else read(args.settings)
    if args.gpu is not None:
        settings['gpu'] = args.gpu
    if args.show_settings:
        print(json.dumps(settings, ensure_ascii=False, indent=2), flush=True)
        return
    # The server's original Phi checkpoint requires its matching PEFT runtime.
    phi_dependencies=Path('/data0/pretrained/tmm_phi4_deps')
    if settings['model']['kind'] == 'phi4' and phi_dependencies.is_dir():
        sys.path.insert(0,str(phi_dependencies))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(settings['gpu'])
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    inputs_dir = ROOT/'cases/emer_qa/inputs'
    records = read(inputs_dir/'model_inputs.json')
    references = read(inputs_dir/'references.json')
    split = read(inputs_dir/'split.json')
    input_hash = hashlib.sha256((inputs_dir/'model_inputs.json').read_bytes()).hexdigest()
    code_files = [Path(__file__), *[(ROOT/'code/tmm/support'/f'emer_{name}.py') for name in ('qa','reader','context_reader','controller','grpo')]]
    code_files.extend(ROOT/path for path in (
        'code/tmm/support/models.py', 'code/tmm/discriminator.py',
        'code/tmm/controller.py', 'code/tmm/support/grpo.py'))
    code_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files}
    identity = {'settings':settings,'input_sha256':input_hash,'split':split,'code_sha256':code_hashes,
                'references_sha256':hashlib.sha256((inputs_dir/'references.json').read_bytes()).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out = ROOT/'cases/emer_qa'/fingerprint[:12]
    out.mkdir(parents=True,exist_ok=True)
    if (out/'settings.json').exists() and read(out/'settings.json') != identity:
        raise ValueError('Experiment identity changed')
    write(out/'settings.json',identity)
    write(out/'code_provenance.json',code_hashes)
    parent=None
    if args.reuse_trained:
        parent=args.reuse_trained.resolve()
        previous=read(parent/'settings.json')
        if any(previous[key] != identity[key] for key in ('settings','input_sha256','split','references_sha256')):
            raise ValueError('Trained-parent data or hyperparameters changed')
        for name in ('reader','context_reader','controller','grpo'):
            key=f'code/tmm/support/emer_{name}.py'
            if previous['code_sha256'][key] != code_hashes[key]:
                raise ValueError('A trained component implementation changed')
        for key in ('code/tmm/support/models.py','code/tmm/discriminator.py',
                    'code/tmm/controller.py','code/tmm/support/grpo.py'):
            if previous['code_sha256'].get(key) != code_hashes[key]:
                raise ValueError('Native model or optimization implementation changed; train a fresh run')
        for name in ('reader','reader_outputs','asr','discriminator','predictions','controller.pt',
                     'controller_report.json','paired_teachers.pt','training_complete.json'):
            target=parent/name
            if target.exists() and not (out/name).exists():
                (out/name).symlink_to(target.resolve(),target_is_directory=target.is_dir())
        write(out/'trained_parent.json',{'source':str(parent),
            'reason':'Explicit reuse of unchanged trained components, data and hyperparameters; existing native outputs retained.'})
    def status(stage, **extra):
        value={'stage':stage,'time_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
               'output':str(out),'fingerprint':fingerprint,'model':settings['model']['name'],
               'model_kind':settings['model']['kind'],**extra}
        write(out/'status.json',value)
        write(ROOT/'cases/emer_qa/status.json',value)
        print(json.dumps(value,ensure_ascii=False),flush=True)
    try:
        if args.stage == 'evaluate':
            required = [out/'discriminator/discriminator.pt', out/'controller.pt', out/'reader/reader_summary.json']
            if settings['reader'].get('mode','source_supervised') == 'source_supervised':
                required.append(out/'reader/reader_bundle.pt')
            for record in records:
                key = record['model_visible']['id']
                required.extend((out/'asr'/(key+'.json'),out/'reader_outputs'/(key+'.json')))
            missing = [str(path.relative_to(out)) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError('Evaluation requires completed weights and saved ASR/Reader outputs; '
                                        'run --stage train first. Missing: '+', '.join(missing))
        import numpy as np
        import torch
        from tmm.support.models import Backbone
        from tmm.support.emer_qa import EmerQA
        from tmm.support.emer_reader import source_reader, read_transcript
        from tmm.support.emer_context_reader import GroundedContextReader
        from tmm.support import emer_controller as ec
        from tmm.controller import Controller
        from tmm.support.emer_grpo import train as train_discriminator
        torch.set_num_threads(settings['cpu_threads'])
        torch.manual_seed(settings['seed'])
        np.random.seed(settings['seed'])
        status('loading_foundation')
        backbone=Backbone(settings['model'],settings['protocol'],'Assess independently which listed emotions are supported.')
        qa=EmerQA(backbone,max_image_pixels=settings['max_image_pixels'])
        reader_config=dict(settings['reader'])
        reader_mode=reader_config.pop('mode','source_supervised')
        reader = None
        if args.stage == 'asr':
            pass
        elif reader_mode == 'grounded_context':
            status('grounded_context_reader')
            reader=GroundedContextReader(qa,reader_config)
            backbone.reader=reader
            summary = {'fingerprint':reader.fingerprint,'settings':reader.config,
                'mode':reader_mode,'training':'none; frozen foundation contextual likelihoods',
                'input_scope':'actual ASR text only; no emotion question, options, or references',
                'span_basis':'exact deterministic source clauses, not learned localization',
                'scope':'EMER qualitative method adaptation; original clinical source-supervised Reader preserved',
                'reference_scope':'no independent human EMER evidence-span annotations'}
            if args.stage == 'evaluate':
                if read(out/'reader/reader_summary.json')['fingerprint'] != reader.fingerprint:
                    raise ValueError('Saved grounded Reader settings/foundation changed')
            else:
                write(out/'reader/reader_summary.json',summary)
        elif reader_mode == 'source_supervised':
            if args.stage == 'evaluate':
                from tmm.reader import ReaderRuntime
                bundle = torch.load(out/'reader/reader_bundle.pt',map_location='cpu',weights_only=False)
                reader = ReaderRuntime(backbone,bundle,include_relation_probabilities=True)
                backbone.reader = reader
                if read(out/'reader/reader_summary.json')['fingerprint'] != reader.fingerprint:
                    raise ValueError('Saved source Reader checkpoint changed')
            else:
                status('source_reader_training')
                reader=source_reader(backbone,out/'reader',settings['seed'],**reader_config)
        else:
            raise ValueError('Unknown EMER Reader mode')
        items, partitions, evidence = {}, {}, {}
        for index, record in enumerate(records):
            item = dict(record['model_visible'])
            item['frames']=[{k:f[k] for k in ('id','path','time')} for f in item['frames']]
            # ASR always uses the same foundation, before any QA adaptation.
            # Retain it across non-ASR implementation repairs.
            asr_key=hashlib.sha256(json.dumps({'audio':item['audio'],'model':settings['model'],
                'prompt':'native_verbatim_json_transcription_v1','max_tokens':settings['asr_max_new_tokens']},sort_keys=True).encode()).hexdigest()
            asr_path=ROOT/'cases/emer_qa/asr_cache'/(asr_key+'.json')
            saved_asr = out/'asr'/(item['id']+'.json')
            if saved_asr.exists():
                asr=read(saved_asr)
            elif args.stage == 'evaluate':
                raise FileNotFoundError('Saved evaluation ASR is missing: '+str(saved_asr))
            elif asr_path.exists():
                asr=read(asr_path)
            else:
                status('transcribing_original_audio',record=item['id'],index=index+1,total=len(records))
                asr=qa.transcribe(item,max_new_tokens=settings['asr_max_new_tokens'])
                write(asr_path,asr)
            if args.stage != 'evaluate':
                write(saved_asr,asr)
            if not asr['text'].strip():
                raise ValueError('Empty actual transcript: '+item['id'])
            item['transcript']=asr['text']
            items[item['id']]=item
            partitions[item['id']]=record['split']
            if args.stage == 'asr':
                continue
            epath=out/'reader_outputs'/(item['id']+'.json')
            if epath.exists():
                saved=read(epath)
                if saved['reader_fingerprint'] != reader.fingerprint:
                    raise ValueError('Reader checkpoint changed')
                er=saved['records']
            else:
                if args.stage == 'evaluate':
                    raise FileNotFoundError('Saved evaluation Reader output is missing: '+str(epath))
                status('reading_observed_transcript',record=item['id'],index=index+1,total=len(records))
                if asr.get('no_speech') or asr.get('unreliable'):
                    er=[]
                elif reader_mode == 'grounded_context':
                    er=reader.read(item)
                else:
                    views=qa.observed_views(item,[])
                    for view in views:
                        view.update(event='speech',speaker='unknown',source=item['audio']['path'],available=True)
                    er=read_transcript(reader,{'views':views})
                write(epath,{'reader_fingerprint':reader.fingerprint,'records':er,
                             'reader_mode':reader_mode,'native_read_log':getattr(reader,'last_read_log',None),
                             'scope':'Machine-predicted context from native ASR; no EMER evidence-span gold'})
            # Canonical JSON types make resumed GRPO signatures identical to fresh records.
            evidence[item['id']]=json.loads(json.dumps(er,ensure_ascii=False))
        if args.stage in ('asr','reader'):
            status('asr_complete' if args.stage == 'asr' else 'reader_complete')
            return
        train_ids=[key for key in items if partitions[key]=='train']
        dev_ids=[key for key in items if partitions[key]=='dev']
        eval_ids=[key for key in items if partitions[key]=='eval']
        status('loading_discriminator' if args.stage == 'evaluate' else 'emotion_discriminator_grpo')
        if parent or args.stage == 'evaluate':
            from tmm.discriminator import _attach, _load_state
            from tmm.support.emer_grpo import DEFAULTS, ALGORITHM
            from tmm.support.grpo import signature
            saved=torch.load(out/'discriminator/discriminator.pt',map_location='cpu',weights_only=False)
            cfg=DEFAULTS | settings['discriminator']
            if (saved['config'] != cfg or saved['report']['train_ids'] != sorted(train_ids)
                    or signature((ALGORITHM,cfg,saved['state_dict'])) != saved['fingerprint']):
                raise ValueError('Completed discriminator weights/configuration changed')
            if read(out/'reader/reader_summary.json')['fingerprint'] != reader.fingerprint:
                raise ValueError('The source Reader changed')
            _attach(backbone,cfg)
            _load_state(backbone,saved['state_dict'])
            backbone.discriminator_bundle=saved
            backbone.discriminator_fingerprint=saved['fingerprint']
            grpo_report={**saved['report'],'reused_unchanged_training_from':str(parent or out)}
        else:
            grpo_report=train_discriminator(qa,[items[k] for k in train_ids],
                {k:references[k]['labels'] for k in train_ids},{k:evidence[k] for k in train_ids},
                out/'discriminator',settings['discriminator'])
        if (grpo_report.get('actual_updates',0) < int(settings['discriminator']['max_updates'])
                or grpo_report.get('parameter_delta_norm',0) <= 0):
            raise ValueError('Discriminator training is incomplete; completed nonzero GRPO updates are required')
        write(out/'discriminator_report.json',grpo_report)
        backbone.model.eval().requires_grad_(False)
        def score(key,acquired,with_reader=True,renderer=0):
            branch='reader' if with_reader else 'plain'
            tag='T'+''.join(sorted(acquired))
            path=out/'predictions'/key/f'{tag}_{branch}_r{renderer}.json'
            if path.exists():
                return read(path)
            value=qa.score(items[key],acquired,evidence[key] if with_reader else [],renderer=renderer,capture_context=True)
            write(path,value)
            return value
        def candidates(key,acquired):
            item=items[key]
            return [{'id':key+':'+m,'modality':m,'start':0.,'stop':item['duration'],'event':'speech','available':True}
                    for m in ('A','V') if m not in acquired]
        def encoded(key,acquired,prediction,with_reader=True):
            return ec.encode_state(qa.observed_views(items[key],acquired),evidence[key] if with_reader else [],
                prediction['controller_context'],prediction['p'],remaining_fraction=(2-len(acquired))/2)
        cpath=out/'controller.pt'
        if cpath.exists():
            controller=Controller.from_state_dict(torch.load(cpath,map_location='cpu',weights_only=True))
        else:
            if args.stage == 'evaluate':
                raise FileNotFoundError('Evaluation requires the completed Controller checkpoint: '+str(cpath))
            pairs={'train':[],'dev':[]}
            for index,key in enumerate(train_ids+dev_ids):
                status('paired_emotion_teachers',record=key,index=index+1,total=len(train_ids+dev_ids))
                states={}
                for acquired in ([],['A'],['V'],['A','V']):
                    states[tuple(acquired)]=[score(key,acquired,True,r) for r in range(3)]
                for acquired in ([],['A'],['V']):
                    before=states[tuple(acquired)]
                    enc=encoded(key,acquired,before[0])
                    for candidate in candidates(key,acquired):
                        after=states[tuple(sorted(acquired+[candidate['modality']]))]
                        pairs[partitions[key]].append(ec.make_pair(key,partitions[key],enc,candidate,
                            [v['p'] for v in before],[v['p'] for v in after],references[key]['labels'],settings['controller']))
            torch.save(pairs,out/'paired_teachers.pt')
            status('controller_fitting',train_pairs=len(pairs['train']),dev_pairs=len(pairs['dev']))
            controller,report=ec.fit(pairs['train'],pairs['dev'],settings['controller'])
            torch.save(controller.state_dict(),cpath)
            write(out/'controller_report.json',report)
        controller.eval().requires_grad_(False)
        if args.stage != 'evaluate':
            write(out/'training_complete.json',{'model':settings['model'],'fingerprint':fingerprint,
                'reader_fingerprint':reader.fingerprint,
                'discriminator_fingerprint':backbone.discriminator_fingerprint,
                'grpo_actual_updates':grpo_report['actual_updates'],
                'train_ids':sorted(train_ids),'dev_ids':sorted(dev_ids),
                'controller_config':settings['controller']})
        if args.stage == 'train':
            status('training_complete',grpo_actual_updates=grpo_report['actual_updates'])
            return
        def metrics(key,prediction):
            y=np.asarray(references[key]['labels'])
            p=np.asarray(prediction['p'])
            pred=p>=.5
            tp=int((pred & (y==1)).sum())
            denom=int(pred.sum()+y.sum())
            return {'brier':float(np.mean((p-y)**2)),'sample_f1':2*tp/denom if denom else 1.,
                    'exact_match':bool(np.array_equal(pred,y)),'answer':prediction['answer']}
        def trajectory(key,with_reader):
            acquired,trace=[],[]
            for _ in range(3):
                prediction=score(key,acquired,with_reader)
                enc=encoded(key,acquired,prediction,with_reader)
                decision=ec.choose(controller,enc,candidates(key,acquired),settings['controller'])
                trace.append({'acquired':list(acquired),'prediction':prediction,'decision':decision})
                if decision['stop']:
                    break
                chosen=next(c for c in candidates(key,acquired) if c['id']==decision['action'])
                acquired=sorted(acquired+[chosen['modality']])
            return {'with_reader':with_reader,'trace':trace,'acquired':acquired,
                    'final':trace[-1]['prediction'],'cost':sum(ec.candidate_cost(
                        {'id':key+':'+m,'modality':m,'start':0.,'stop':items[key]['duration']},settings['controller']) for m in acquired)}
        results=[]
        for index,key in enumerate(eval_ids):
            case_path=out/'evaluation'/(key+'.json')
            if case_path.exists():
                case=read(case_path)
            else:
                status('case_search',record=key,index=index+1,total=len(eval_ids))
                full=trajectory(key,True)
                no_reader=trajectory(key,False)
                seed=score(key,[],False)
                reader_seed=score(key,[],True)
                same=score(key,full['acquired'],False)
                measured={name:metrics(key,pred) for name,pred in
                    [('seed',seed),('reader_seed',reader_seed),('full',full['final']),
                     ('without_reader',no_reader['final']),('same_acquired_without_reader',same)]}
                mg=measured
                reader_gain=mg['same_acquired_without_reader']['brier']-mg['full']['brier']
                controller_gain=mg['reader_seed']['brier']-mg['full']['brier']
                reader_f1_gain=mg['full']['sample_f1']-mg['same_acquired_without_reader']['sample_f1']
                controller_f1_gain=mg['full']['sample_f1']-mg['reader_seed']['sample_f1']
                joint=bool(evidence[key] and 'V' in full['acquired'] and mg['full']['exact_match']
                           and not mg['seed']['exact_match'] and reader_f1_gain>1e-6 and controller_f1_gain>1e-6
                           and mg['full']['sample_f1']>mg['without_reader']['sample_f1']+1e-6)
                case={'id':key,'partition':'eval','reference':references[key],
                      'model_visible':items[key],'reader_records':evidence[key],
                      'seed_prediction':seed,'reader_seed_prediction':reader_seed,
                      'full':full,'without_reader':no_reader,'same_acquired_without_reader':same,
                      'metrics':measured,'reader_brier_gain_at_fixed_acquisition':reader_gain,
                      'controller_brier_gain_from_reader_seed':controller_gain,'joint_candidate':joint,
                      'reader_f1_gain_at_fixed_acquisition':reader_f1_gain,
                      'controller_f1_gain_from_reader_seed':controller_f1_gain,
                      'limits':['Qualitative EMER emotion QA adaptation, not a clinical screening benchmark.',
                                'Reader is a frozen foundation contextual likelihood adaptation; no human EMER span gold.',
                                'Exact sentence spans are deterministic; confidence is choice likelihood, not calibrated accuracy.',
                                'The previously examined case pool is qualitative development, not an untouched benchmark test.',
                                'No-reader uses the same fitted Controller without Reader inputs; this is not a retrained ablation.',
                                'The constrained 0/1 outputs come from actual native option logits.']}
                write(case_path,case)
            results.append({'id':key,**case['metrics'],'joint_candidate':case['joint_candidate'],
                            'reader_gain':case['reader_brier_gain_at_fixed_acquisition'],
                            'controller_gain':case['controller_brier_gain_from_reader_seed'],
                            'acquired':case['full']['acquired'],'records':len(case['reader_records'])})
            write(out/'case_index.json',{'cases':results,'actual_evaluated':len(results),'maximum':len(eval_ids)})
        winners=sorted((r for r in results if r['joint_candidate']),
            key=lambda r:(r['full']['exact_match'],r['full']['sample_f1'],min(r['reader_gain'],r['controller_gain'])),reverse=True)
        supported_ids=[]
        for row in winners[:3]:
            key=row['id']
            case=read(out/'evaluation'/(key+'.json'))
            # Remaining source is scored once for a post-hoc STOP/selection diagnostic.
            full=case['full']
            cf=[]
            for candidate in candidates(key,full['acquired']):
                acquired=sorted(full['acquired']+[candidate['modality']])
                pred=score(key,acquired,True)
                cf.append({'added_modality':candidate['modality'],'prediction':pred,'metrics':metrics(key,pred),
                           'scope':'post-hoc counterfactual evaluation; never available to the inference Controller'})
            case['remaining_acquisition_counterfactuals']=cf
            first_actions=[]
            for modality in ('A','V'):
                pred=score(key,[modality],True)
                first_actions.append({'modality':modality,'prediction':pred,'metrics':metrics(key,pred),
                                      'scope':'matched one-acquisition counterfactual; not used by the Controller'})
            case['one_acquisition_counterfactuals']=first_actions
            chosen=full['trace'][0]['decision']['action'].rsplit(':',1)[-1]
            chosen_loss=next((r['metrics']['brier'] for r in first_actions if r['modality']==chosen),None)
            alternative_loss=next((r['metrics']['brier'] for r in first_actions if r['modality']!=chosen),None)
            choice_helps=(chosen_loss is not None and alternative_loss is not None and chosen_loss+1e-4<alternative_loss)
            stop_helps=any(r['metrics']['brier']>case['metrics']['full']['brier']+1e-4 for r in cf)
            case['controller_selection_supported']=bool(choice_helps or stop_helps)
            case['controller_selection_basis']={'chosen_first_better_than_other':choice_helps,
                                                 'rejecting_remaining_source_avoids_harm':stop_helps}
            case['joint_label_correction_supported']=bool(case['joint_candidate'])
            case['both_modules_supported']=bool(case['joint_candidate'] and case['controller_selection_supported'])
            if case['both_modules_supported']:
                supported_ids.append(key)
            write(out/'selected_cases'/(key+'.json'),case)
        aggregate={name:{'mean_sample_f1':float(np.mean([r[name]['sample_f1'] for r in results])),
                         'mean_brier':float(np.mean([r[name]['brier'] for r in results])),
                         'exact_matches':sum(r[name]['exact_match'] for r in results)}
                   for name in ('seed','reader_seed','full','without_reader','same_acquired_without_reader')}
        all_cases=[read(out/'evaluation'/(key+'.json')) for key in eval_ids]
        measured=evaluation_metrics(all_cases)
        write(out/'metrics.json',{'model':settings['model'],'fingerprint':fingerprint,**measured})
        summary={'fingerprint':fingerprint,'scope':'bounded EMER qualitative QA adaptation',
                 'model':settings['model'],
                 'train_videos':len(train_ids),'dev_videos':len(dev_ids),'evaluated_videos':len(results),
                 'joint_candidates':len(winners),'selected_ids':[r['id'] for r in winners[:3]],
                 'both_modules_supported_ids':supported_ids,
                 'grpo_actual_updates':grpo_report.get('actual_updates'),
                 'selection_criterion':'nonempty Reader; native V acquired; full exact correction of Seed error; full sample F1 strictly above Reader seed, same-acquired plain, and no-Reader trajectory',
                 'aggregate':aggregate,'evaluation_metrics':measured['conditions'],
                 'reader_mode':reader_mode,'results':results}
        write(out/'summary.json',summary)
        status('complete',joint_candidates=len(winners),selected_ids=summary['selected_ids'])
    except Exception as exc:
        status('failed',error=str(exc),traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
