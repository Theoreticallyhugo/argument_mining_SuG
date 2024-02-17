"""
functions for saving metadata of models
"""
import os
import json


def save_meta(path: str, data: dict):
    with open(os.path.join(os.curdir, path, "meta.json"), "w") as w:
        for key in data.keys():
            try:
                data[key]["number"] = int(data[key]["number"])
            except:
                pass
        json.dump(data, w)
