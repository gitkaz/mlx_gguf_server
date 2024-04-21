import os
import math


def get_model_size(model_path: str):
    model_size = 0
    if os.path.isfile(model_path):
        model_size = os.path.getsize(model_path)

    elif os.path.isdir(model_path):
        for file in os.scandir(model_path):
            if os.path.isfile(file):
                model_size += os.path.getsize(file)

    model_size_in_gb = model_size / (1024 * 1024 * 1024)
    model_size_in_gb = math.floor(model_size_in_gb)
    return model_size_in_gb


def create_model_list():
    models ={}
    for f in os.listdir("models"):
        if os.path.isdir(os.path.join("models", f)) or f.endswith(".gguf"):
            model_name = f
            model_path = os.path.join('models', model_name)
            model_size = get_model_size(model_path)
            models[model_name] = {'path': model_path, 'size': model_size }

    return models


