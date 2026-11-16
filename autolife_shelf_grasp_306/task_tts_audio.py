"""选择真实扬声器声卡，不依赖声卡序号或已停用的桌面声音服务器。"""
from pathlib import Path
import re
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
