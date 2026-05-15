"""Tests for prompt_gate.py — convergence-driven LSH training gate."""

import torch
import pytest
from packr.zpackr_layer import ZPackRLinear
from packr.prompt_gate import should_skip_backward


class TestConvergenceGate:
    def test_gate_fires_when_all_attenuated(self):
        lin = torch.nn.Linear(64, 32, bias=False)
        zpl = ZPackRLinear.from_linear(lin)
        # Set all rows to max attenuation
        zpl._atten_byte = torch.full((64,), 255, dtype=torch.uint8, device=zpl.delta_salient.device)
        layers = [("test", zpl)]
        assert should_skip_backward(layers, threshold=0.9)

    def test_gate_does_not_fire_when_any_novel(self):
        lin = torch.nn.Linear(64, 32, bias=False)
        zpl = ZPackRLinear.from_linear(lin)
        zpl._atten_byte = torch.full((64,), 255, dtype=torch.uint8, device=zpl.delta_salient.device)
        zpl._atten_byte[0] = 0  # one row is novel
        layers = [("test", zpl)]
        assert not should_skip_backward(layers, threshold=0.9)

    def test_gate_returns_false_when_no_attenuation(self):
        lin = torch.nn.Linear(64, 32, bias=False)
        zpl = ZPackRLinear.from_linear(lin)
        zpl._atten_byte = torch.zeros(64, dtype=torch.uint8, device=zpl.delta_salient.device)
        layers = [("test", zpl)]
        assert not should_skip_backward(layers, threshold=0.9)

    def test_gate_returns_true_with_empty_layers(self):
        assert should_skip_backward([], threshold=0.9)

    def test_gate_with_custom_threshold(self):
        lin = torch.nn.Linear(64, 32, bias=False)
        zpl = ZPackRLinear.from_linear(lin)
        # 128/255 ≈ 0.5
        zpl._atten_byte = torch.full((64,), 128, dtype=torch.uint8, device=zpl.delta_salient.device)
        layers = [("test", zpl)]
        # threshold=0.8, max attn=0.5 → don't skip
        assert not should_skip_backward(layers, threshold=0.8)
        # threshold=0.4, max attn=0.5 → skip
        assert should_skip_backward(layers, threshold=0.4)
