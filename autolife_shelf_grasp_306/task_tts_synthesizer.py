"""独立进程加载本地Matcha和声码器，避免与视觉Torch环境互相污染。"""
import argparse
from pathlib import Path
import sys
import wave


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--vocoder',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--volume',type=float,default=.65);a=parser.parse_args()
    import numpy as np
    import sherpa_onnx
    m=Path(a.model)
    model=next(iter(sorted(m.glob('model*.onnx'))),None)
    if model is None:raise RuntimeError('缺少Matcha声学模型')
    config=sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(acoustic_model=str(model),vocoder=a.vocoder,
            lexicon=str(m/'lexicon.txt'),tokens=str(m/'tokens.txt'),dict_dir=str(m/'dict')),
        num_threads=2,debug=False,provider='cpu'),max_num_sentences=1)
    if not config.validate():raise RuntimeError('本地TTS配置校验失败')
    tts=sherpa_onnx.OfflineTts(config);audio=tts.generate(sys.stdin.read(),sid=0,speed=a.speed)
    samples=(np.clip(np.asarray(audio.samples,dtype=float)*a.volume,-1,1)*32767).astype('<i2')
    with wave.open(a.output,'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(audio.sample_rate);f.writeframes(samples.tobytes())

if __name__=='__main__':main()
