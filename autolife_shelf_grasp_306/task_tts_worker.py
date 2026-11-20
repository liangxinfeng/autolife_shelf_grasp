"""独立中文离线播报进程；文字缓存，停止时终止整个播放进程组。"""
import argparse,hashlib,json,os,subprocess,sys,tempfile,wave
from pathlib import Path


def synthesis_environment(assets):
    """Use installed sherpa, with an optional bundle for this Python ABI."""
    env=dict(os.environ)
    version=f'python{sys.version_info.major}.{sys.version_info.minor}'
    bundled=Path(assets)/'lib'/version/'site-packages'
    if bundled.is_dir():
        existing=[part for part in env.get('PYTHONPATH','').split(os.pathsep) if part]
        env['PYTHONPATH']=os.pathsep.join(dict.fromkeys([str(bundled),*existing]))
    return env

def cache_key(text,speed=1.):
    text=str(text).strip()
    if not text or len(text)>500:raise ValueError('播报文字应为1～500字')
    return hashlib.sha256(json.dumps([text,float(speed),'matcha-steps3-hifigan-v2'],ensure_ascii=False).encode()).hexdigest()


def voice_duration(path):
    """Validate complete, nonempty PCM before admitting a file to the cache."""
    with wave.open(str(path),'rb') as stream:
        frames=stream.getnframes();rate=stream.getframerate()
        width=stream.getsampwidth();channels=stream.getnchannels()
        if frames<=0 or rate<=0 or channels<=0 or stream.getcomptype()!='NONE':
            raise ValueError('合成语音没有有效 PCM 数据')
        size=0;audible=False
        while True:
            data=stream.readframes(65536)
            if not data:break
            size+=len(data)
            audible=audible or any(value!=(128 if width==1 else 0) for value in data)
        if size!=frames*width*channels or not audible:
            raise ValueError('合成语音数据不完整或为空')
        return frames/rate


def prepare_voice(assets,path,text,speed):
    try:
        return voice_duration(path),True
    except (OSError,EOFError,wave.Error,ValueError):
        pass
    # Concurrent requests may synthesize the same sentence. They publish only
    # complete files and never share or delete another worker's scratch file.
    with tempfile.NamedTemporaryFile(prefix=path.stem+'.',suffix='.tmp.wav',dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name)
    try:
        command=[sys.executable,str(Path(__file__).with_name('task_tts_synthesizer.py')),
            '--model',str(assets/'models/matcha-icefall-zh-baker'),'--vocoder',str(assets/'models/hifigan_v2.onnx'),
            '--output',str(temporary),'--speed',str(speed),'--volume','1']
        subprocess.run(command,input=text,text=True,check=True,env=synthesis_environment(assets),timeout=90)
        duration=voice_duration(temporary)
        os.replace(temporary,path)
        return duration,False
    finally:
        temporary.unlink(missing_ok=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--synthesize-only',action='store_true');args=a.parse_args()
    root=Path(args.root);payload=json.loads(sys.stdin.read());text=payload['text'];speed=float(payload.get('speed',1.))
    if not .5<=speed<=2:raise ValueError('语速需在0.5～2之间')
    assets=root/'models/task_tts';cache=root/'data/voices';cache.mkdir(parents=True,exist_ok=True)
    path=cache/(cache_key(text,speed)+'.wav')
    duration,cached=prepare_voice(assets,path,text,speed)
    playback=None
    if not args.synthesize_only:
        sys.path.insert(0,str(Path(__file__).parent))
        from task_tts_audio import playback_command
        command,env=playback_command(path,payload.get('audio_device','auto'))
        subprocess.run(command,env=env,check=True,timeout=duration+8)
        playback=command[0]
    print(json.dumps(dict(ok=True,path=str(path),duration=duration,cache_hit=cached,playback=playback)))

if __name__=='__main__':main()
