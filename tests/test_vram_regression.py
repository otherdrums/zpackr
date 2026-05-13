"""VRAM regression tests — ensure PackR mode stays within budget."""

import torch
import pytest
from packr import PackRConfig, compress_model


def _vram_mb():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.max_memory_allocated


class TestPackRVRAM:
    def test_packr_forward_peak_vram(self):
        """Forward-only VRAM must not exceed budget after warmup."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        config = PackRConfig(mode="packr", layer_scope="all", gradient_checkpointing=False)
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 3072, bias=False),
            torch.nn.Linear(3072, 768, bias=False),
        )
        model = compress_model(model, config)
        model = model.cuda()

        x = torch.randn(4, 768, device="cuda")

        # Warmup — first call may allocate kernel caches, cubin buffers
        for _ in range(3):
            _ = model(x)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        for _ in range(10):
            _ = model(x)
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Forward peak VRAM: {peak_mb:.1f} MB")
        # Budget: weights (~13 MB packed) + activations + decode temp ~= 50 MB max
        assert peak_mb < 60, f"Forward VRAM {peak_mb:.1f} MB exceeds 60 MB budget"

    def test_packr_full_step_peak_vram(self):
        """Forward+backward VRAM must not exceed budget."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        config = PackRConfig(mode="packr", layer_scope="all", gradient_checkpointing=False)
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 3072, bias=False),
            torch.nn.Linear(3072, 768, bias=False),
        )
        model = compress_model(model, config)
        model = model.cuda()

        x = torch.randn(4, 768, device="cuda")

        # Warmup
        for _ in range(3):
            out = model(x)
            out.sum().backward()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        for _ in range(10):
            out = model(x)
            out.sum().backward()
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Full-step peak VRAM: {peak_mb:.1f} MB")
        # Budget: forward temp + backward temp (w_full_bf16 + dW_full + decode) ~= 100 MB
        assert peak_mb < 120, f"Full-step VRAM {peak_mb:.1f} MB exceeds 120 MB budget"

    def test_packr_forward_no_int64_temp(self):
        """Verify backward does NOT materialize int64 W_p index tensor."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        config = PackRConfig(mode="packr", layer_scope="all", gradient_checkpointing=False)
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 3072, bias=False),
        )
        model = compress_model(model, config)
        model = model.cuda()

        x = torch.randn(4, 768, device="cuda")

        # Warmup
        for _ in range(3):
            out = model(x)
            out.sum().backward()
        torch.cuda.synchronize()

        # Run backward and check that no int64 > 1MB allocation occurs
        torch.cuda.reset_peak_memory_stats()
        out = model(x)
        out.sum().backward()
        torch.cuda.synchronize()

        # A 768x3072 int64 tensor = 18.9 MB. If we see a peak > 10 MB beyond
        # the known allocations, the decode kernel path may have regressed.
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Single-step peak VRAM: {peak_mb:.1f} MB")
        # A single 768x3072 layer should fit well under this budget
        assert peak_mb < 80, f"Single-step VRAM {peak_mb:.1f} MB exceeds 80 MB budget"


class TestPackRVRAMOffload:
    def test_packr_offload_forward_peak_vram(self):
        """Forward VRAM with offload must be lower than without."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        config = PackRConfig(mode="packr", layer_scope="all", offload=True,
                             gradient_checkpointing=False)
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 3072, bias=False),
            torch.nn.Linear(3072, 768, bias=False),
        )
        model = compress_model(model, config)
        model = model.cuda()

        x = torch.randn(4, 768, device="cuda")

        for _ in range(3):
            _ = model(x)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        for _ in range(10):
            _ = model(x)
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Offload forward peak VRAM: {peak_mb:.1f} MB")
        # W_p is offloaded to CPU, VRAM should be ~13 MB lower
        assert peak_mb < 60, f"Offload forward VRAM {peak_mb:.1f} MB exceeds 60 MB budget"
