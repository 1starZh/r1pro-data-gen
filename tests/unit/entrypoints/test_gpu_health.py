from __future__ import annotations

from r1pro_data_gen.infrastructure import gpu_health as module


def test_probe_accepts_the_requested_driver_backed_gpu(monkeypatch) -> None:
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    class Completed:
        returncode = 0
        stdout = "0,RTX A6000,550.54,49140\n6,RTX A6000,550.54,49140\n"
        stderr = ""

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: Completed())

    report = module.probe_gpu_health(6)

    assert report["healthy"] is True
    assert report["selected_gpu"]["index"] == 6
    assert report["selected_gpu"]["memory_total_mib"] == "49140"


def test_probe_fails_closed_when_driver_query_fails(monkeypatch) -> None:
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    class Completed:
        returncode = 9
        stdout = "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver."
        stderr = ""

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: Completed())

    report = module.probe_gpu_health(6)

    assert report["healthy"] is False
    assert report["gpus"] == []
    assert report["reason"] == "nvidia-smi could not communicate with the NVIDIA driver"


def test_probe_rejects_invalid_physical_gpu_id_without_running_nvidia_smi(monkeypatch) -> None:
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("nvidia-smi must not run for an invalid GPU id")

    monkeypatch.setattr(module.subprocess, "run", fail_if_called)

    report = module.probe_gpu_health(-1)

    assert report["healthy"] is False
    assert "non-negative integer" in report["reason"]
    assert called is False
