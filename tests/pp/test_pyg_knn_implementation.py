import torch

from omicverse._optional import normalize_torch_device
from omicverse.pp import pyg_knn_implementation as pyg_knn


def test_optional_device_helper_import_has_no_torch_or_settings_side_effects():
    import subprocess
    import sys

    code = """
import sys
from omicverse._optional import normalize_torch_device
assert callable(normalize_torch_device)
assert "torch" not in sys.modules
assert "omicverse._settings" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_normalize_torch_device_adds_current_cuda_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    device = normalize_torch_device("cuda")

    assert device == torch.device("cuda:0")


def test_normalize_torch_device_preserves_explicit_cuda_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    device = normalize_torch_device("cuda:1")

    assert device == torch.device("cuda:1")


def test_normalize_torch_device_uses_current_cuda_for_none(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)

    device = normalize_torch_device()

    assert device == torch.device("cuda:2")


def test_normalize_torch_device_integer_selects_cuda_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    device = normalize_torch_device(1)

    assert device == torch.device("cuda:1")


def test_autotune_chunk_receives_indexed_cuda_device(monkeypatch):
    seen = {}

    def fake_mem_get_info(device):
        seen["device"] = device
        return 2 << 30, 4 << 30

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", fake_mem_get_info)

    device = normalize_torch_device("cuda")
    pyg_knn._autotune_chunk(100, 10, device)

    assert seen["device"] == torch.device("cuda:0")


def test_chunked_knn_normalizes_device_before_autotune(monkeypatch):
    seen = {}

    def fake_autotune(n_samples, n_features, device, headroom_gb=1.0):
        seen["device"] = device
        return n_samples

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(pyg_knn, "_autotune_chunk", fake_autotune)
    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda self, *args, **kwargs: self,
    )

    X = torch.zeros((2, 2), dtype=torch.float32)
    pyg_knn._chunked_knn_l2(X, k=1, device=torch.device("cuda"))

    assert seen["device"] == torch.device("cuda:0")


def test_normalize_torch_device_falls_back_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    device = normalize_torch_device("cuda", fallback_to_cpu=True)

    assert device == torch.device("cpu")


def test_normalize_torch_device_preserves_cuda_without_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    device = normalize_torch_device("cuda")

    assert device == torch.device("cuda")
