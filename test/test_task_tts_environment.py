"""TTS environment and immutable cache protocol; no audio device is opened."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import pytest
from autolife_shelf_grasp_306 import task_tts_worker as worker


def test_installed_sherpa_path_survives_without_a_private_bundle(tmp_path,monkeypatch):
    monkeypatch.setenv('PYTHONPATH','/installed/sherpa'+os.pathsep+'/application')
    (tmp_path/'lib/python0.0/site-packages').mkdir(parents=True)
    result=worker.synthesis_environment(tmp_path)
    assert result['PYTHONPATH']==os.environ['PYTHONPATH']


def test_optional_bundle_must_match_current_python_and_preserves_existing_paths(tmp_path,monkeypatch):
    version=f'python{sys.version_info.major}.{sys.version_info.minor}'
    package=tmp_path/'lib'/version/'site-packages';package.mkdir(parents=True)
    monkeypatch.setenv('PYTHONPATH',os.pathsep.join(['/installed/sherpa',str(package)]))
    assert worker.synthesis_environment(tmp_path)['PYTHONPATH'].split(os.pathsep)==[str(package),'/installed/sherpa']


def test_first_synthesis_then_cache_hit_never_calls_audio_playback(tmp_path,monkeypatch,capsys):
    calls=[]
    def synthesize(command,**kwargs):
        assert 'task_tts_synthesizer.py' in command[1]
        path=Path(command[command.index('--output')+1]);calls.append(command)
        with wave.open(str(path),'wb') as stream:
            stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000)
            stream.writeframes(b'\x64\x00\x9c\xff'*800)
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(worker.subprocess,'run',synthesize)
    monkeypatch.setattr(sys,'argv',['tts','--root',str(tmp_path),'--synthesize-only'])
    payload=json.dumps(dict(text='语音缓存验证',speed=1.))
    results=[];stamp=None
    for _ in range(2):
        monkeypatch.setattr(sys,'stdin',io.StringIO(payload));worker.main()
        result=json.loads(capsys.readouterr().out);results.append(result)
        current=Path(result['path']).stat().st_mtime_ns
        if stamp is not None:assert current==stamp
        stamp=current
    assert len(calls)==1 and results[0]['cache_hit'] is False and results[1]['cache_hit'] is True
    assert results[0]['path']==results[1]['path'] and results[1]['duration']>0


def test_failed_synthesis_cannot_publish_a_cache_entry(tmp_path,monkeypatch):
    def fail(command,**kwargs):
        Path(command[command.index('--output')+1]).write_bytes(b'partial wav')
        raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(worker.subprocess,'run',fail)
    monkeypatch.setattr(sys,'argv',['tts','--root',str(tmp_path),'--synthesize-only'])
    monkeypatch.setattr(sys,'stdin',io.StringIO(json.dumps(dict(text='失败测试'))))
    with pytest.raises(subprocess.CalledProcessError):worker.main()
    assert not list((tmp_path/'data/voices').glob('*.wav'))


def write_voice(path):
    with wave.open(str(path),'wb') as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000)
        stream.writeframes(b'\x64\x00\x9c\xff'*800)


def test_concurrent_same_text_uses_independent_temporary_files(tmp_path,monkeypatch):
    ready=Barrier(2);paths=[]
    def synthesize(command,**kwargs):
        path=Path(command[command.index('--output')+1]);paths.append(path)
        write_voice(path)
        ready.wait(timeout=5)
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(worker.subprocess,'run',synthesize)
    target=tmp_path/'cached.wav'
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker.prepare_voice,tmp_path,target,'并发合成',1.) for _ in range(2)]
        results=[future.result(timeout=10) for future in futures]
    assert len(set(paths))==2 and all(not path.exists() for path in paths)
    assert results==[(.1,False),(.1,False)]
    assert worker.voice_duration(target)==.1
    assert list(tmp_path.glob('*.wav'))==[target]


@pytest.mark.parametrize('corruption',['bad_header','truncated','silent'])
def test_damaged_cache_is_rebuilt_before_use(tmp_path,monkeypatch,corruption):
    target=tmp_path/'cached.wav';write_voice(target)
    if corruption=='bad_header':target.write_bytes(b'broken')
    elif corruption=='truncated':target.write_bytes(target.read_bytes()[:-2])
    else:
        with wave.open(str(target),'wb') as stream:
            stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000)
            stream.writeframes(b'\x00'*3200)
    calls=[]
    def synthesize(command,**kwargs):
        calls.append(command);write_voice(Path(command[command.index('--output')+1]))
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(worker.subprocess,'run',synthesize)
    assert worker.prepare_voice(tmp_path,target,'重建语音',1.)==(.1,False)
    assert worker.prepare_voice(tmp_path,target,'重建语音',1.)==(.1,True)
    assert len(calls)==1 and list(tmp_path.glob('*.wav'))==[target]


def test_success_exit_with_invalid_wave_cannot_publish_cache(tmp_path,monkeypatch):
    def invalid(command,**kwargs):
        Path(command[command.index('--output')+1]).write_bytes(b'not a wave')
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(worker.subprocess,'run',invalid)
    with pytest.raises((wave.Error,EOFError,ValueError)):
        worker.prepare_voice(tmp_path,tmp_path/'cached.wav','损坏的合成结果',1.)
    assert not list(tmp_path.glob('*.wav'))
