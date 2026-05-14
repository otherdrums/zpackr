"""Model-level checkpoint save/load for ZPackR.

LZ4-compressed delta serialization — no dictionary state needed.
"""

import os
import torch


def save_zpackr_checkpoint(model, path: str):
    from .zpackr_layer import ZPackRLinear

    os.makedirs(path, exist_ok=True)
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, ZPackRLinear):
            module.save_checkpoint(os.path.join(path, str(layer_idx)))
            layer_idx += 1

    meta = {
        "version": 2,
        "num_layers": layer_idx,
        "layer_names": [
            name for name, m in model.named_modules()
            if isinstance(m, ZPackRLinear)
        ],
    }
    torch.save(meta, os.path.join(path, "meta.pt"))


def load_zpackr_checkpoint(model, path: str):
    from .zpackr_layer import ZPackRLinear

    meta = torch.load(os.path.join(path, "meta.pt"), weights_only=True)

    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, ZPackRLinear):
            if layer_idx >= meta["num_layers"]:
                break
            restored = ZPackRLinear.load_checkpoint(os.path.join(path, str(layer_idx)))
            module.delta_salient.data = restored.delta_salient.data
            module.block_mask.copy_(restored.block_mask)
            module._full_delta = restored._full_delta
            module._zstd_delta = restored._zstd_delta
            if module.bias is not None and restored.bias is not None:
                module.bias.data = restored.bias.data
            layer_idx += 1

    return model
