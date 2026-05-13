"""Shared test fixtures for packr/ZPackR."""

import os
import sys
import pytest
import torch

# Ensure packr is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def super_dict():
    from zpackr.super_dict import load_super_dict
    return load_super_dict()


@pytest.fixture
def weight_dict():
    from zpackr.zstd_dict import WeightDict
    return WeightDict(max_entries=16384)
