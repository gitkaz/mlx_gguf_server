import json
from .generate import generate
import numpy as np

def run(params, queue):
    try:
        audio_data = generate(params)
        if isinstance(audio_data, bytes):
            # 如果已经是 bytes 数据则直接发送
            queue.put(audio_data)
        elif isinstance(audio_data, str):
            # 如果是字符串数据也直接发送
            queue.put(audio_data)
        else:
            # 其他类型视为错误
            queue.put(json.dumps({"error":"generate.generate return not bytes or np.ndarray or str"}))
    except Exception as e:
        queue.put(json.dumps({"error": str(e)})) # 在错误发生时也将错误内容传回