"""Versioned machine-readable benchmark manifests."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tokserve.benchmarking.config import BenchmarkConfig
from tokserve.benchmarking.environment import (
    EnvironmentMetadata,
    capture_environment,
    interpreter_metadata,
)

BENCHMARK_SCHEMA_VERSION = "tokserve.benchmark.v1"


@dataclass(frozen=True)
class BenchmarkManifest:
    """Configuration and provenance recorded before measurement."""

    schema_version: str
    created_at_utc: str
    config: BenchmarkConfig
    environment: EnvironmentMetadata
    interpreter: dict[str, str]
    measurements: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        """Validate the manifest schema."""

        if self.schema_version != BENCHMARK_SCHEMA_VERSION:
            raise ValueError("unsupported benchmark schema version")

        if not self.created_at_utc.endswith("Z"):
            raise ValueError("created_at_utc must use UTC Z notation")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe manifest representation."""

        return {
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "config": self.config.to_dict(),
            "environment": self.environment.to_dict(),
            "interpreter": dict(self.interpreter),
            "measurements": [dict(measurement) for measurement in self.measurements],
        }


def create_manifest(
    config: BenchmarkConfig,
    *,
    environment: EnvironmentMetadata | None = None,
    created_at_utc: str | None = None,
) -> BenchmarkManifest:
    """Create an empty measurement manifest for a validated run."""

    captured_environment = (
        capture_environment(config) if environment is None else environment
    )
    timestamp = (
        datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        if created_at_utc is None
        else created_at_utc
    )

    return BenchmarkManifest(
        schema_version=BENCHMARK_SCHEMA_VERSION,
        created_at_utc=timestamp,
        config=config,
        environment=captured_environment,
        interpreter=interpreter_metadata(),
        measurements=(),
    )


def write_manifest(
    manifest: BenchmarkManifest,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one formatted JSON manifest to its configured destination."""

    output_path = manifest.config.output_path

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"benchmark output already exists: {output_path}")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(
            manifest.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path
