"""Model-level checkpoint save/load for ZPackR v2.0.

Enables reversible training eras: loading a checkpoint from an earlier
epoch restores the WeightDict state, weights, and salience at that point.
Blocks that were "learned" later become novel again — the model temporarily
"forgets" everything after the checkpoint era.
"""

import os
import torch


def save_zpackr_checkpoint(model, path: str):
    """Save zstd_weights, WeightDict state, and block_mask for every ZPackRLinear layer.

    Args:
        model: nn.Module with ZPackRLinear layers.
        path: Base path for checkpoint files (e.g. "checkpoints/epoch_0").
              Produces path/N.zstd, path/N.mask, path/N.wd.*, etc.
    """
    from .zpackr_layer import ZPackRLinear

    os.makedirs(path, exist_ok=True)

    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, ZPackRLinear):
            layer_path = os.path.join(path, str(layer_idx))
            module.save_checkpoint(layer_path)
            layer_idx += 1

    # Save metadata (layer mapping)
    meta = {
        "version": 1,
        "num_layers": layer_idx,
        "layer_names": [
            name for name, m in model.named_modules()
            if isinstance(m, ZPackRLinear)
        ],
    }
    torch.save(meta, os.path.join(path, "meta.pt"))


def load_zpackr_checkpoint(model, path: str):
    """Restore model to a previous training era.

    Decompresses zstd_weights, restores WeightDict state, recomputes
    salience.  Blocks that were "learned" at the checkpoint's era become
    novel again under the restored dict.

    Args:
        model: nn.Module with ZPackRLinear layers.
        path: Base path for checkpoint files.
    """
    from .zpackr_layer import ZPackRLinear

    meta = torch.load(os.path.join(path, "meta.pt"), weights_only=True)

    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, ZPackRLinear):
            if layer_idx >= meta["num_layers"]:
                break

            layer_path = os.path.join(path, str(layer_idx))
            restored = ZPackRLinear.load_checkpoint(layer_path, module.weight_dict)

            # Swap the module in-place
            module.delta_salient.data = restored.delta_salient.data
            module.block_mask.copy_(restored.block_mask)
            module._full_delta = restored._full_delta
            module._zstd_delta = restored._zstd_delta
            if module.bias is not None and restored.bias is not None:
                module.bias.data = restored.bias.data
            module._acc = None

            layer_idx += 1

    return model
