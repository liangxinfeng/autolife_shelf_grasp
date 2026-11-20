"""默认复用当前用户的桌面音频服务；无服务时按真实 ALSA 声卡回退。"""
from pathlib import Path
import os,re,shutil,subprocess
AppError = ValueError


def resolve_audio_device(configured='auto',asound=Path('/proc/asound')):
    value=str(configured).strip()
    if value and value!='auto':return value
    candidates=set()
    try:
        for line in (asound/'pcm').read_text().splitlines():
            # HDMI通常是显示器音频；不把它猜作机身扬声器。
            if 'playback' not in line.lower() or 'hdmi' in line.lower():continue
            match=re.match(r'^(\d+)-\d+:',line)
            if match:
                card=(asound/('card'+str(int(match[1])))/'id').read_text().strip()
                if re.fullmatch(r'[A-Za-z0-9_]+',card):candidates.add('sysdefault:CARD='+card)
    except OSError as exc:raise AppError('无法读取机载声音设备：'+str(exc)) from exc
    if len(candidates)!=1:
        raise AppError('未发现唯一机载扬声器，请在TTS参数audio_device中指定设备；发现：'+(', '.join(sorted(candidates)) or '无'))
    return next(iter(candidates))


def playback_command(path,configured='auto',asound=Path('/proc/asound')):
    """Choose once before playback; a failed playback must not seize the card."""
    value=str(configured).strip();env=dict(os.environ)
    if not value or value=='auto':
        # SSH/worker launchers may omit XDG_RUNTIME_DIR. Reuse only this user's
        # runtime directory; never connect to another logged-in user's session.
        if not env.get('XDG_RUNTIME_DIR') and hasattr(os,'getuid'):
            uid=os.getuid();runtime=Path('/run/user')/str(uid)
            try:
                if runtime.is_dir() and runtime.stat().st_uid==uid:
                    env['XDG_RUNTIME_DIR']=str(runtime)
            except OSError:pass
        if not env.get('PULSE_SERVER') and env.get('XDG_RUNTIME_DIR'):
            socket=Path(env['XDG_RUNTIME_DIR'])/'pulse/native'
            if socket.is_socket():env['PULSE_SERVER']='unix:'+str(socket)
        for probe,player in ((['pactl','info'],'paplay'),(['pw-cli','info','0'],'pw-play')):
            # An explicit server prevents libpulse autospawning a new daemon
            # when no desktop session is running.
            if player=='paplay' and not env.get('PULSE_SERVER'):continue
            if not shutil.which(probe[0]) or not shutil.which(player):continue
            try:
                result=subprocess.run(probe,env=env,capture_output=True,text=True,timeout=2)
            except (OSError,subprocess.TimeoutExpired):continue
            if result.returncode==0:return [player,str(path)],env
    return ['aplay','-q','-D',resolve_audio_device(value,asound),str(path)],env
