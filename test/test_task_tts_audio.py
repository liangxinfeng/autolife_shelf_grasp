"""Audio transport selection; all subprocesses and device data are fake."""
import io
import json
from pathlib import Path
import subprocess
import sys
import wave
import pytest
from autolife_shelf_grasp_306 import task_tts_audio as audio,task_tts_worker as worker


def alsa_cards(tmp_path):
    (tmp_path/'card1').mkdir();(tmp_path/'card1/id').write_text('PCH')
    (tmp_path/'pcm').write_text('01-00: ALC Analog : playback 1 : capture 1\n00-03: HDMI : playback 1\n')
    return tmp_path


def test_explicit_device_wins_without_probing_desktop(monkeypatch,tmp_path):
    monkeypatch.setattr(audio.subprocess,'run',lambda *a,**k:pytest.fail('explicit device must not probe'))
    command,env=audio.playback_command('voice.wav','plughw:CARD=External,DEV=0',tmp_path)
    assert command==['aplay','-q','-D','plughw:CARD=External,DEV=0','voice.wav']


@pytest.mark.parametrize('service,player',[('pactl','paplay'),('pw-cli','pw-play')])
def test_auto_uses_available_desktop_service_instead_of_raw_card(monkeypatch,tmp_path,service,player):
    monkeypatch.setenv('PULSE_SERVER','unix:/fake/session/pulse/native')
    monkeypatch.setattr(audio.shutil,'which',lambda name:'/usr/bin/'+name)
    calls=[]
    def probe(command,**kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command,0 if command[0]==service else 1)
    monkeypatch.setattr(audio.subprocess,'run',probe)
    command,env=audio.playback_command('voice.wav','auto',tmp_path/'absent_alsa')
    assert command==[player,'voice.wav'] and calls[-1][0]==service


@pytest.mark.parametrize('unavailable',['no_tools','not_running','timeout'])
def test_no_desktop_service_keeps_existing_alsa_selection(monkeypatch,tmp_path,unavailable):
    monkeypatch.setenv('PULSE_SERVER','unix:/fake/session/pulse/native')
    monkeypatch.setattr(audio.shutil,'which',lambda name:None if unavailable=='no_tools' else '/usr/bin/'+name)
    def probe(command,**kwargs):
        if unavailable=='timeout':raise subprocess.TimeoutExpired(command,2)
        return subprocess.CompletedProcess(command,1)
    monkeypatch.setattr(audio.subprocess,'run',probe)
    command,env=audio.playback_command('voice.wav','auto',alsa_cards(tmp_path))
    assert command==['aplay','-q','-D','sysdefault:CARD=PCH','voice.wav']


def test_no_service_or_card_reports_failure(monkeypatch,tmp_path):
    monkeypatch.setattr(audio.shutil,'which',lambda name:None)
    (tmp_path/'pcm').write_text('00-03: HDMI : playback 1\n')
    with pytest.raises(ValueError,match='未发现唯一机载扬声器'):
        audio.playback_command('voice.wav','auto',tmp_path)


def test_ssh_missing_runtime_uses_only_current_users_owned_directory(monkeypatch,tmp_path):
    uid=tmp_path.stat().st_uid
    monkeypatch.setattr(audio.os,'getuid',lambda:uid,raising=False)
    runtime=tmp_path/'users'/str(uid);runtime.mkdir(parents=True)
    monkeypatch.setattr(audio,'Path',lambda value:tmp_path/'users' if value=='/run/user' else Path(value))
    monkeypatch.delenv('XDG_RUNTIME_DIR',raising=False)
    monkeypatch.delenv('PULSE_SERVER',raising=False)
    monkeypatch.setattr(Path,'is_socket',lambda path:path==runtime/'pulse/native')
    monkeypatch.setattr(audio.shutil,'which',lambda name:'/usr/bin/'+name)
    def probe(command,**kwargs):
        assert kwargs['env']['XDG_RUNTIME_DIR']==str(runtime)
        assert kwargs['env']['PULSE_SERVER']=='unix:'+str(runtime/'pulse/native')
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(audio.subprocess,'run',probe)
    command,env=audio.playback_command('voice.wav','auto',tmp_path/'absent_alsa')
    assert command==['paplay','voice.wav'] and env['XDG_RUNTIME_DIR']==str(runtime)


def test_missing_desktop_socket_does_not_autospawn_pulse(monkeypatch,tmp_path):
    monkeypatch.delenv('PULSE_SERVER',raising=False)
    monkeypatch.setenv('XDG_RUNTIME_DIR',str(tmp_path/'no_session'))
    monkeypatch.setattr(audio.shutil,'which',lambda name:'/usr/bin/'+name)
    def probe(command,**kwargs):
        assert command[0]!='pactl'
        return subprocess.CompletedProcess(command,1)
    monkeypatch.setattr(audio.subprocess,'run',probe)
    command,env=audio.playback_command('voice.wav','auto',alsa_cards(tmp_path))
    assert command==['aplay','-q','-D','sysdefault:CARD=PCH','voice.wav']


@pytest.mark.parametrize('failed',[False,True])
def test_worker_uses_selected_desktop_route_once_and_does_not_retry_raw_device(monkeypatch,tmp_path,capsys,failed):
    target=tmp_path/'data/voices'/(worker.cache_key('音频路由')+'.wav');target.parent.mkdir(parents=True)
    with wave.open(str(target),'wb') as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000)
        stream.writeframes(b'\x64\x00\x9c\xff'*800)
    monkeypatch.setattr(sys,'argv',['tts','--root',str(tmp_path)])
    monkeypatch.setattr(sys,'stdin',io.StringIO(json.dumps(dict(text='音频路由'))))
    monkeypatch.setitem(sys.modules,'task_tts_audio',audio)
    monkeypatch.setattr(audio,'playback_command',lambda *a:(['paplay',str(target)],{'XDG_RUNTIME_DIR':'/runtime'}))
    calls=[]
    def playback(command,**kwargs):
        calls.append(command)
        assert kwargs['env']=={'XDG_RUNTIME_DIR':'/runtime'} and kwargs['check']
        if failed:raise subprocess.CalledProcessError(1,command)
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(worker.subprocess,'run',playback)
    if failed:
        with pytest.raises(subprocess.CalledProcessError):worker.main()
        assert not capsys.readouterr().out
    else:
        worker.main();result=json.loads(capsys.readouterr().out)
        assert result['playback']=='paplay' and result['cache_hit']
    assert calls==[['paplay',str(target)]]
