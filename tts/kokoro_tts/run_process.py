import json
from .generate import generate
import numpy as np

def run(params, queue):
    try:
        audio_data = generate(params)
        if isinstance(audio_data, bytes):
            #  すでにbytesデータの場合はそのまま送信
            queue.put(audio_data)
        elif isinstance(audio_data, str):
             #  文字列データの場合もそのまま
            queue.put(audio_data)
        else:
            #  それ以外の場合エラー
            queue.put(json.dumps({"error":"generate.generate return not bytes or np.ndarray or str"}))
    except Exception as e:
        queue.put(json.dumps({"error": str(e)})) # エラー時にもエラー内容を伝える