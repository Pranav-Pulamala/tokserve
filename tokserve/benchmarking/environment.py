"""Environment metadata capture for reproducible benchmarks."""

import importlib.metadata
import platform
import sys
from dataclasses import dataclass

import torch

import tokserve
from tokserve.benchmarking.config import BenchmarkConfig
from tokserve.engine.device import resolve_device


def _package_version(package: str) -> str | None:
    """Return an installed distribution version when available."""

    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class EnvironmentMetadata:
    """Software and hardware provenance for one benchmark run."""

    python_version: str
    operating_system: str
    machine: str
    processor: str
    tokserve_version: str
    torch_version: str
    torch_cuda_version: str | None
    triton_version: str | None
    requested_device: str
    resolved_device: str
    cuda_available: bool
    cuda_device_name: str | None
    cuda_device_count: int
    cuda_capability: str | None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe environment metadata."""

        return {
            "python_version": self.python_version,
            "operating_system": self.operating_system,
            "machine": self.machine,
            "processor": self.processor,
            "tokserve_version": self.tokserve_version,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "triton_version": self.triton_version,
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "cuda_available": self.cuda_available,
            "cuda_device_name": self.cuda_device_name,
            "cuda_device_count": self.cuda_device_count,
            "cuda_capability": self.cuda_capability,
        }


def capture_environment(
    config: BenchmarkConfig,
) -> EnvironmentMetadata:
    """Capture runtime metadata without running inference."""

    device = resolve_device(config.device)
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_device_name: str | None = None
    cuda_capability: str | None = None

    if device.type == "cuda":
        device_index = device.index if device.index is not None else 0
        cuda_device_name = torch.cuda.get_device_name(device_index)
        major, minor = torch.cuda.get_device_capability(device_index)
        cuda_capability = f"{major}.{minor}"

    return EnvironmentMetadata(
        python_version=platform.python_version(),
        operating_system=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        tokserve_version=tokserve.__version__,
        torch_version=torch.__version__,
        torch_cuda_version=(
            str(torch.version.cuda) if torch.version.cuda is not None else None
        ),
        triton_version=_package_version("triton"),
        requested_device=config.device,
        resolved_device=str(device),
        cuda_available=cuda_available,
        cuda_device_name=cuda_device_name,
        cuda_device_count=cuda_device_count,
        cuda_capability=cuda_capability,
    )


def interpreter_metadata() -> dict[str, str]:
    """Return explicit interpreter implementation details."""

    return {
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }
