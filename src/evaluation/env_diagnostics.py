"""Environment diagnostics for Colab/local inference checks (no model download)."""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict


def _bytes_to_gb(num_bytes: int | float | None) -> float | None:
    if num_bytes is None:
        return None
    return round(float(num_bytes) / (1024 ** 3), 2)


def collect_environment_report() -> Dict[str, Any]:
    """Collect runtime environment information relevant to inference suitability."""
    report: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": None,
        "transformers_version": None,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "gpu_compute_capability": None,
        "system_ram_gb": None,
        "cuda_device_count": 0,
    }

    try:
        import psutil  # type: ignore

        report["system_ram_gb"] = _bytes_to_gb(psutil.virtual_memory().total)
    except Exception as exc:  # pragma: no cover - environment dependent
        report["system_ram_gb"] = None
        report["system_ram_error"] = str(exc)

    try:
        import transformers  # type: ignore

        report["transformers_version"] = getattr(transformers, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - environment dependent
        report["transformers_version"] = None
        report["transformers_error"] = str(exc)

    try:
        import torch  # type: ignore

        report["torch_version"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if report["cuda_available"] and report["cuda_device_count"] > 0:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            report["gpu_name"] = props.name
            report["gpu_vram_gb"] = _bytes_to_gb(getattr(props, "total_memory", None))
            report["gpu_compute_capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:  # pragma: no cover - environment dependent
        report["torch_error"] = str(exc)

    return report


def assess_inference_suitability(report: Dict[str, Any]) -> Dict[str, Any]:
    """Return a simple suitability assessment for Gemma 4 E2B baseline inference."""
    reasons = []
    suitable = True

    if not report.get("torch_version"):
        suitable = False
        reasons.append("PyTorch is not installed.")
    if not report.get("transformers_version"):
        suitable = False
        reasons.append("Transformers is not installed.")

    cuda = bool(report.get("cuda_available"))
    vram = report.get("gpu_vram_gb")
    ram = report.get("system_ram_gb")

    if cuda:
        if vram is not None and vram < 8:
            reasons.append(
                f"Detected GPU VRAM is {vram} GB; inference may be tight. >=12 GB is more comfortable."
            )
        elif vram is not None:
            reasons.append(f"CUDA GPU detected ({report.get('gpu_name')}, {vram} GB VRAM).")
        else:
            reasons.append("CUDA GPU detected but VRAM could not be determined.")
    else:
        suitable = False
        reasons.append("CUDA GPU not available; Gemma 4 E2B inference in Colab CPU-only runtimes is likely impractical.")
        if ram is not None and ram < 16:
            reasons.append(f"System RAM is {ram} GB; too low for reliable CPU inference.")

    if not reasons:
        reasons.append("Environment appears minimally sufficient for inference.")

    return {"suitable_for_inference": suitable, "reasons": reasons}


def print_environment_report(report: Dict[str, Any], assessment: Dict[str, Any]) -> None:
    """Pretty-print diagnostics and suitability assessment."""
    print("=== Environment Diagnostics ===")
    for key in [
        "python_version",
        "platform",
        "torch_version",
        "transformers_version",
        "cuda_available",
        "cuda_device_count",
        "gpu_name",
        "gpu_vram_gb",
        "gpu_compute_capability",
        "system_ram_gb",
    ]:
        print(f"{key}: {report.get(key)}")
    for key in ["system_ram_error", "transformers_error", "torch_error"]:
        if report.get(key):
            print(f"{key}: {report.get(key)}")

    print("\n=== Inference Suitability ===")
    print(f"suitable_for_inference: {assessment.get('suitable_for_inference')}")
    for reason in assessment.get("reasons", []):
        print(f"- {reason}")
