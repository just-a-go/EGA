"""Prepare original EMER media and a fixed, bounded qualitative adaptation split."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'cases/emer_qa/inputs'
REMOTE_ROOT = '/data0/codefile/wangweiqi/TMM/cases/emer_qa/inputs'
PRIORITY = [f'sample_{number:08d}.mp4' for number in (1371, 2518, 1557, 1416, 1636, 2789, 1411)]
EVAL_EXTENSION = 12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-root', type=Path, default=Path('/data0/data/IntentBench/videos'),
                        help='Directory containing the original source videos')
    parser.add_argument('--remote', help='Optional SSH alias for fetching missing source videos')
    args = parser.parse_args()
    rows = [r for r in json.loads((OUT/'source_qa.json').read_text(encoding='utf-8-sig'))
            if r.get('source') == 'EMER']
    by_video = {r['video']: r for r in rows}
    others = sorted(set(by_video)-set(PRIORITY), key=lambda s: hashlib.sha256(('27|'+s).encode()).hexdigest())
    base_eval = PRIORITY + others[32:45]
    # Append the first unused videos under the original seed/hash ordering.
    # Selection never reads answers, explanations, Reader output or QA scores.
    extension = others[45:45+EVAL_EXTENSION]
    partitions = {'train': others[:24], 'dev': others[24:32], 'eval': base_eval+extension}
    videos = OUT/'videos'
    media = OUT/'media'
    videos.mkdir(parents=True, exist_ok=True)
    media.mkdir(exist_ok=True)
    selected = sum(partitions.values(), [])
    missing = [name for name in selected if not (videos/name).is_file()]
    for name in missing:
        source = args.video_root/name
        if source.is_file():
            shutil.copy2(source, videos/name)
    missing = [name for name in selected if not (videos/name).is_file()]
    if missing:
        if not args.remote:
            raise FileNotFoundError(
                f'{len(missing)} source videos missing. Set --video-root DIR or --remote SSH_ALIAS.')
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                        *[f'{args.remote}:{args.video_root.as_posix()}/{name}' for name in missing],
                        str(videos)], check=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    from PIL import Image
    references, inputs, split_rows = {}, [], []
    for split, names in partitions.items():
        for name in names:
            key = Path(name).stem
            source = videos/name
            wave = media/(key+'.wav')
            if not wave.exists():
                subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-i', str(source),
                                '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', '-y', str(wave)], check=True)
            import wave as wave_module
            with wave_module.open(str(wave)) as handle:
                duration = handle.getnframes()/handle.getframerate()
            frame_rows = []
            for j, fraction in enumerate((.15, .5, .85)):
                second = duration*fraction
                frame = media/f'{key}_{j}.png'
                if not frame.exists():
                    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-ss', str(second), '-i', str(source),
                                    '-frames:v', '1', '-y', str(frame)], check=True)
                with Image.open(frame) as picture:
                    dimensions = list(picture.size)
                frame_rows.append({'id': f'{key}:V:{j}', 'path': REMOTE_ROOT+'/media/'+frame.name,
                                   'time': second, 'dimensions': dimensions})
            row = by_video[name]
            visible = {'id': key, 'question': row['problem'], 'options': row['options'],
                       'transcript': '', 'audio': {'id':key+':A','path':REMOTE_ROOT+'/media/'+wave.name,'start':0.,'stop':duration},
                       'frames':frame_rows, 'duration':duration, 'language':'Chinese'}
            inputs.append({'id':key,'split':split,'model_visible':visible})
            references[key] = {'source':'EMER', 'video':name, 'answer':row['answer'],
                               'options':row['options'], 'problem':row['problem'], 'problem_type':row['problem_type'],
                               'labels':[int(chr(65+i) in {s.strip() for s in row['answer'].split(',')}) for i in range(len(row['options']))],
                               'video_server_path':args.video_root.as_posix().rstrip('/')+'/'+name,
                               'video_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                               'reference_scope':'emotion QA labels; not clinical diagnosis or evidence-span gold'}
            split_rows.append({'id':key,'split':split,'video':name,'priority_case_candidate':name in PRIORITY})
    (OUT/'model_inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'references.json').write_text(json.dumps(references,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'split.json').write_text(json.dumps({'seed':27,'train':24,'dev':8,'eval':len(partitions['eval']),'records':split_rows,
        'selection':'Original 24 train, 8 dev and 20 eval unchanged; append the first 12 previously unused videos by SHA256("27|" + video filename)',
        'scope':'bounded qualitative adaptation; not a subject-disjoint benchmark result',
        'no_inference_fields':['answer','labels','clean_list','solution','context','think']},ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'eval_extension.json').write_text(json.dumps({
        'schema':'emer-fixed-case-search-extension-v1', 'seed':27,
        'source_qa_sha256':hashlib.sha256((OUT/'source_qa.json').read_bytes()).hexdigest(),
        'official_emer_videos':len(by_video), 'previously_unused_videos':len(others[45:]),
        'selection':'Sort previously unused video filenames by SHA256("27|" + filename); take the first 12 before inference.',
        'selection_inputs':['source video filename', 'fixed seed 27'],
        'excluded_selection_inputs':['answer','labels','clean_list','solution','context','think','model outputs','evaluation metrics'],
        'scope':'Additional qualitative case-search samples; not new training data or an independent benchmark evaluation.',
        'unchanged_train':partitions['train'], 'unchanged_dev':partitions['dev'],
        'unchanged_base_eval':base_eval, 'new_eval_videos':extension,
        'media':'Original videos; original-resolution PNG frames at 15%, 50%, 85% of decoded audio duration; mono 16 kHz PCM waveform.',
        'new_video_sha256':{Path(name).stem:references[Path(name).stem]['video_sha256'] for name in extension},
        'reference_boundary':'Answers are evaluation-only emotion labels, not Reader evidence-span gold.',
    },ensure_ascii=False,indent=2),encoding='utf-8')
    print('Prepared',len(inputs),'original clips:', {k:len(v) for k,v in partitions.items()},flush=True)


if __name__ == '__main__':
    main()
