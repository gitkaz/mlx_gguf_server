import io
import os
import scipy.io.wavfile
import numpy as np
import torch
from kokoro import KPipeline, KModel
from pydub import AudioSegment


def generate(params):
    # config = os.environ["KOKORO_CONFIG"]
    print(f"debug: kokoro_generate: params = {params}")
    model = KModel()
    pipeline = KPipeline(lang_code=params['lang_code'], model=model)

    generator = pipeline(
        params['text'], voice=params['voice'], speed=params['speed'], split_pattern=params['split_pattern']
    )
    all_audio = []
    sample_rate = 24000
    for i, (gs, ps, audio) in enumerate(generator):
        if not isinstance(audio, np.ndarray):
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().detach().numpy()
            else:
                raise TypeError(f"Unexpected audio type: {type(audio)}")
        all_audio.append(audio)

    combined_audio = np.concatenate(all_audio)

    # 使用空的 BytesIO 来保存 WAV 与 MP3 数据
    wav_data = io.BytesIO()    
    mp3_data = io.BytesIO() 
    scipy.io.wavfile.write(wav_data, sample_rate, combined_audio)  # 使用 scipy.io.wavfile 将 WAV 数据写入内存
    wav_data.seek(0)  # 将文件指针归位到开头
    audio_segment = AudioSegment.from_wav(wav_data)  # 使用 pydub.AudioSegment 读取内存中的 WAV 数据

    audio_segment.export(mp3_data, format="mp3", bitrate="32k")  # 转换为 MP3 格式

    # 返回 MP3 数据的 bytes
    return mp3_data.getvalue()