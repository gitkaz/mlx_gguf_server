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

    # 空のByteIOを用意（WAV用とMP3用）
    wav_data = io.BytesIO()    
    mp3_data = io.BytesIO() 

    scipy.io.wavfile.write(wav_data, sample_rate, combined_audio)  # scipy.io.wavfile を使って WAV データをメモリ上に保存
    wav_data.seek(0)  # ファイルポインタを先頭に戻す
    audio_segment = AudioSegment.from_wav(wav_data) # pydub.AudioSegment を使ってメモリ上の WAV データを読み込む

    audio_segment.export(mp3_data, format="mp3", bitrate="32k")  # MP3形式に変換

    # MP3データをBytesで返す
    return mp3_data.getvalue()