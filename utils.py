import math, os, random, json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

def seed_everything(seed: int = 0):
    """设全家桶种子（python、numpy、torch、cuda）+ 打开确定性模式"""
    import os, random
    import numpy as np
    import torch
    
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")  # 有些CUDA需要
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

# 别名函数，更明确的命名
def set_all_seeds(seed: int):
    """设置所有随机种子，确保完全可重现性"""
    seed_everything(seed)

def create_worker_init_fn(base_seed: int):
    """创建DataLoader worker初始化函数，确保多进程可重现"""
    def _worker_init(worker_id):
        import numpy as np, random, torch
        s = base_seed + worker_id
        np.random.seed(s)
        random.seed(s) 
        torch.manual_seed(s)
    return _worker_init

def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config_file(path: str | None):
    """
    Load a config file (YAML or JSON). Returns {} if path is None.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("Reading YAML requires PyYAML: pip install pyyaml") from e
        return yaml.safe_load(text) or {}
    raise ValueError(f"Unsupported config format: {suffix}. Use .json or .yaml")


def timestamp_str() -> str:
    """Return local timestamp string like YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
