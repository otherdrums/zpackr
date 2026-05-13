"""ZPackR model interface — zstd-native prompt handling and training helpers."""

import torch
from .super_dict import ZPackRSuperDict


def prompt_zstd(model, zstd_bytes: bytes):
    """Run inference on a zstd-compressed prompt.

    Decompresses via Super Dict, tokenizes, runs forward, returns
    (output, compression_ratio).
    """
    if not hasattr(model, "super_zstd") or not hasattr(model, "_tokenizer"):
        raise RuntimeError("Model must have super_zstd and _tokenizer for zstd prompts")

    sup = model.super_zstd
    tok = model._tokenizer

    text = sup.decompress_to_text(zstd_bytes)
    ratio = len(zstd_bytes) / max(len(sup.compress(text.encode("utf-8"))), 1)

    tokens = tok(text, return_tensors="pt", truncation=True, padding=True)
    tokens = {k: v.to(next(model.parameters()).device) for k, v in tokens.items()}

    model.eval()
    with torch.no_grad():
        output = model(**tokens)

    return output, ratio


def prompt_zstd_with_learning(model, zstd_bytes: bytes, threshold: float = 2.0):
    """Run inference and trigger training if prompt is novel.

    If the Super Dict ratio is low (novel prompt), runs forward AND backward,
    training the delta.  Returns (output, ratio, trained: bool).
    """
    if not hasattr(model, "super_zstd") or not hasattr(model, "_tokenizer"):
        raise RuntimeError("Model must have super_zstd and _tokenizer for zstd prompts")

    sup = model.super_zstd
    tok = model._tokenizer

    text = sup.decompress_to_text(zstd_bytes)
    text_bytes = text.encode("utf-8")
    ratio = sup.compress(text_bytes)

    tokens = tok(text, return_tensors="pt", truncation=True, padding=True)
    tokens = {k: v.to(next(model.parameters()).device) for k, v in tokens.items()}

    model.train()
    output = model(**tokens)
    trained = False

    if ratio < threshold:
        loss = output.loss
        loss.backward()
        trained = True

    return output, ratio, trained


def export_model(model, output_path: str = None):
    """Export ZPackR model as a standard HuggingFace model.

    Merges base_W + delta for every ZPackRLinear layer, replaces with
    nn.Linear.  Returns a plain model ready for HuggingFace save.
    """
    from .zpackr_layer import ZPackRLinear
    import torch.nn as nn

    for name, module in model.named_modules():
        if isinstance(module, ZPackRLinear):
            merged = module.export_merged()  # [out, in] for nn.Linear
            new_lin = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
            new_lin.weight.data.copy_(merged)
            if module.bias is not None:
                new_lin.bias.data.copy_(module.bias.data)

            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_lin)

    if output_path:
        model.save_pretrained(output_path)
    return model
