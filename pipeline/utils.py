"""Standardization utilities and helper functions."""

import numpy as np
import torch


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y") if not isinstance(v, bool) else v


def parse_hidden_dims(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def fit_standardizer(data):
    mu = data.mean(axis=0)
    std = data.std(axis=0)
    std[std < 1e-8] = 1.0
    return {"mean": mu.astype(np.float32), "std": std.astype(np.float32)}


def apply_standardizer(data, stats):
    if stats is None:
        return data.copy()
    return ((data - stats["mean"]) / stats["std"]).astype(np.float32)


def invert_standardizer(data, stats):
    if stats is None:
        return data
    return (data * stats["std"] + stats["mean"]).astype(np.float32)


def infer_is_discrete_task(name):
    return any(kw.lower() in name.lower() for kw in ["TFBind", "GFP", "UTR", "ChEMBL"])


def is_logit_encoded_discrete(name):
    return any(kw.lower() in name.lower() for kw in ["tfbind"])


def _normalize_01(a):
    lo, hi = a.min(), a.max()
    return np.full_like(a, 0.5) if hi - lo < 1e-12 else (a - lo) / (hi - lo)


def clear_cuda_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
