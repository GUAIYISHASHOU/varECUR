from __future__ import annotations
import json, os, random, yaml
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [to_device(x, device) for x in batch]
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config_file(path_or_none):
    if path_or_none is None:
        return {"common": {}, "train": {}, "model": {}, "runtime": {}, "eval": {}}
    p = Path(path_or_none)
    with open(p, "r", encoding="utf-8") as f:
        if p.suffix.lower() in (".yml", ".yaml"):
            return yaml.safe_load(f)
        return json.load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
