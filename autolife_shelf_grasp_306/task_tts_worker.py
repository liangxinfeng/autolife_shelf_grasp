"""独立中文离线播报进程；文字缓存，停止时终止整个播放进程组。"""
import argparse,hashlib,json,os,subprocess,sys,wave
from pathlib import Path

def cache_key(text,speed=1.):
    text=str(text).strip()
    if not text or len(text)>500:raise ValueError('播报文字应为1～500字')
    return hashlib.sha256(json.dumps([text,float(speed),'matcha-steps3-hifigan-v2'],ensure_ascii=False).encode()).hexdigest()

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--synthesize-only',action='store_true');args=a.parse_args()
    root=Path(args.root);payload=json.loads(sys.stdin.read());text=payload['text'];speed=float(payload.get('speed',1.))
    if not .5<=speed<=2:raise ValueError('语速需在0.5～2之间')
    assets=root/'models/task_tts';cache=root/'data/voices';cache.mkdir(parents=True,exist_ok=True)
    path=cache/(cache_key(text,speed)+'.wav')
    if not path.is_file():
        env=dict(os.environ);env['PYTHONPATH']=str(assets/'lib/python3.12/site-packages')
        command=[sys.executable,str(Path(__file__).with_name('task_tts_synthesizer.py')),
            '--model',str(assets/'models/matcha-icefall-zh-baker'),'--vocoder',str(assets/'models/hifigan_v2.onnx'),
            '--output',str(path.with_suffix('.tmp.wav')),'--speed',str(speed),'--volume','1']
        subprocess.run(command,input=text,text=True,check=True,env=env,timeout=90)
        os.replace(path.with_suffix('.tmp.wav'),path)
    with wave.open(str(path),'rb') as f:duration=f.getnframes()/f.getframerate()
    if not args.synthesize_only:
        sys.path.insert(0,str(Path(__file__).parent))
        from task_tts_audio import resolve_audio_device
        device=resolve_audio_device(payload.get('audio_device','auto'))
        subprocess.run(['aplay','-q','-D',device,str(path)],check=True,timeout=duration+8)
    print(json.dumps(dict(ok=True,path=str(path),duration=duration)))

if __name__=='__main__':main()
