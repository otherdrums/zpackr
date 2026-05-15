"""ZPackR layer patcher — replaces nn.Linear with ZPackRLinear."""

import torch.nn as nn
from packr.offload import OffloadManager
from .zpackr_layer import ZPackRLinear
from .config import ZPackRConfig


def compress_model(model: nn.Module, config: ZPackRConfig = None):
    """Replace nn.Linear layers with ZPackR-compressed equivalents.

    Returns:
        model: nn.Module with ZPackRLinear layers.
    """
    if config is None:
        config = ZPackRConfig()

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not _matches_scope(name, config.layer_scope):
            continue

        zpackr = ZPackRLinear.from_linear(module, hash_interval=config.hash_interval)
        _replace_module(model, name, zpackr)

    if config.gradient_checkpointing:
        _enable_gradient_checkpointing(model)

    if config.offload:
        if next(model.parameters()).is_cpu:
            model.cuda()
        mgr = OffloadManager(prefetch_depth=1)
        zpackr_layers = [(n, m) for n, m in model.named_modules()
                          if isinstance(m, ZPackRLinear)]
        for name, layer in zpackr_layers:
            mgr.register_wp(name, layer.base_W)
        model._offload_manager = mgr

    return model


def _replace_module(model, name, new_module):
    parent = model
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _matches_scope(name: str, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "ffn":
        return _is_ffn(name)
    if scope == "attention":
        return _is_attention(name) and not _is_ffn(name)
    return False


def _is_ffn(name: str) -> bool:
    ffn_markers = ["intermediate", "fc1", "mlp.up", "ffn.up", "dense_h_to_4h",
                   "output.dense", "fc2", "mlp.down", "ffn.down", "dense_4h_to_h"]
    name_lower = name.lower()
    if any(m in name_lower for m in ffn_markers[:5]):
        return True
    if any(m in name_lower for m in ffn_markers[5:]):
        if "attention" not in name_lower:
            return True
    return False


def _is_attention(name: str) -> bool:
    attn_markers = ["query", "key", "value", "q_proj", "k_proj", "v_proj", "o_proj", "out_proj"]
    return any(m in name.lower() for m in attn_markers)


def _enable_gradient_checkpointing(model: nn.Module):
    try:
        model.gradient_checkpointing_enable()
    except AttributeError:
        pass
