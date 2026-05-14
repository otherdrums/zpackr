"""Tests for prompt_gate.py — convergence-driven LZ4 training gate."""

import torch
import pytest
from packr.zpackr_layer import ZPackRLinear
from packr.prompt_gate import should_skip_backward


class TestConvergenceGate:
    def test_gate_fires_when_all_attenuated(self):
        lin = torch.nn.Linear(64, 32, bias=False)
        zpl = ZPackRLinear.from_linear(lin)
        zpl._attenuation_factors = [0.95, 0.92, 0.91]
        layers = [("test", zpl)]
        assert should_skip_backward(layers, threshold=0.9)

    def test_gate_does_not_fire_when_any_novel(self):
        lin = torch.nn.Linear(64, 32, bias=False)
        zpl = ZPackRLinear.from_linear(lin)
        zpl._attenuation_factors = [0.95, 0.50, 0.91]
        layers = [("test", zpl)]
        assert not should_skip_backward(layers, threshold=0.9)

    def test_gate_returns_false_when_no_factors(self):
        lin = torch.nn.Linear(64, 32, bias=False)
        zpl = ZPackRLinear.from_linear(lin)
        layers = [("test", zpl)]
        assert not should_skip_backward(layers, threshold=0.9)

    def test_gate_returns_true_with_empty_layers(self):
        assert should_skip_backward([], threshold=0.9)

    def test_gate_with_custom_threshold(self):
        lin = torch.nn.Linear(64, 32, bias=False)
        zpl = ZPackRLinear.from_linear(lin)
        zpl._attenuation_factors = [0.5, 0.6, 0.7]
        layers = [("test", zpl)]
        # With threshold=0.8, some blocks below → don't skip
        assert not should_skip_backward(layers, threshold=0.8)
        # With threshold=0.4, all blocks above → skip
        assert should_skip_backward(layers, threshold=0.4)
