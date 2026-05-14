"""Shared test fixtures for ZPackR."""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
