"""Validated aggregation of real TokServe benchmark artifacts."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import cast

from tokserve.benchmarking.manifest import BENCHMARK_SCHEMA_VERSION
from tokserve.benchmarking.serving import SERVING_SCHEMA_VERSION

REPORT_SCHEMA_VERSION = "tokserve.benchmark-report.v1"
MINIMUM_P99_SAMPLES = 100

JSONMapping = dict[str, object]


@dataclass(frozen=True)
class DistributionSummary:
    """Summary derived from raw nonnegative samples."""

    sample_count: int
    minimum: float
    median: float
    p95: float
    p99: float | None
    maximum: float
    mean: float
    coefficient_of_variation: float | None

    def to_dict(self) -> JSONMapping:
        """Return a JSON-safe representation."""

        return {
            "sample_count": self.sample_count,
            "minimum": self.minimum,
            "median": self.median,
            "p95": self.p95,
            "p99": self.p99,
            "maximum": self.maximum,
            "mean": self.mean,
            "coefficient_of_variation": self.coefficient_of_variation,
        }


@dataclass(frozen=True)
class ArtifactSummary:
    """Validated summary of one input artifact."""

    path: Path
    schema_version: str
    source_revision: str
    environment: JSONMapping
    config: JSONMapping
    measurements: tuple[JSONMapping, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> JSONMapping:
        """Return a JSON-safe representation."""

        return {
            "path": str(self.path),
            "schema_version": self.schema_version,
            "source_revision": self.source_revision,
            "environment": self.environment,
            "config": self.config,
            "measurements": list(self.measurements),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class WorkloadComparison:
    """One baseline-versus-optimized workload comparison."""

    workload: str
    comparable: bool
    baseline_path: Path
    optimized_path: Path
    baseline_latency: DistributionSummary
    optimized_latency: DistributionSummary
    baseline_tokens_per_second: float | None
    optimized_tokens_per_second: float | None
    latency_speedup: float | None
    throughput_speedup: float | None
    warnings: tuple[str, ...]

    def to_dict(self) -> JSONMapping:
        """Return a JSON-safe representation."""

        return {
            "workload": self.workload,
            "comparable": self.comparable,
            "baseline_path": str(self.baseline_path),
            "optimized_path": str(self.optimized_path),
            "baseline_latency_seconds": self.baseline_latency.to_dict(),
            "optimized_latency_seconds": self.optimized_latency.to_dict(),
            "baseline_tokens_per_second": self.baseline_tokens_per_second,
            "optimized_tokens_per_second": self.optimized_tokens_per_second,
            "latency_speedup": self.latency_speedup,
            "latency_speedup_definition": (
                "baseline median latency / optimized median latency"
            ),
            "throughput_speedup": self.throughput_speedup,
            "throughput_speedup_definition": (
                "optimized aggregate token throughput / "
                "baseline aggregate token throughput"
            ),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class BenchmarkReport:
    """Complete benchmark aggregation report."""

    artifacts: tuple[ArtifactSummary, ...]
    comparisons: tuple[WorkloadComparison, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> JSONMapping:
        """Return the versioned JSON-safe report."""

        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "warnings": list(self.warnings),
            "resume_metrics": [
                _resume_metric(comparison)
                for comparison in self.comparisons
                if comparison.comparable and comparison.latency_speedup is not None
            ],
        }


def percentile(
    samples: tuple[float, ...],
    percentile_value: float,
) -> float:
    """Return a linearly interpolated percentile."""

    if not samples:
        raise ValueError("percentile requires at least one sample")

    if percentile_value < 0.0 or percentile_value > 100.0:
        raise ValueError("percentile must be in [0, 100]")

    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_distribution(
    samples: tuple[float, ...],
) -> DistributionSummary:
    """Summarize one real sample population."""

    if not samples:
        raise ValueError("at least one sample is required")

    if any(not math.isfinite(value) or value < 0.0 for value in samples):
        raise ValueError("samples must be finite and nonnegative")

    sample_mean = mean(samples)
    variation = (
        pstdev(samples) / sample_mean
        if len(samples) > 1 and sample_mean > 0.0
        else None
    )

    return DistributionSummary(
        sample_count=len(samples),
        minimum=min(samples),
        median=median(samples),
        p95=percentile(samples, 95.0),
        p99=(
            percentile(samples, 99.0) if len(samples) >= MINIMUM_P99_SAMPLES else None
        ),
        maximum=max(samples),
        mean=sample_mean,
        coefficient_of_variation=variation,
    )


def load_artifact(path: Path) -> ArtifactSummary:
    """Load and validate one benchmark artifact."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error

    payload = _mapping(raw, "artifact")
    schema_version = _string(payload, "schema_version")

    if schema_version == BENCHMARK_SCHEMA_VERSION:
        return _load_inference_artifact(path, payload)

    if schema_version == SERVING_SCHEMA_VERSION:
        return _load_serving_artifact(path, payload)

    raise ValueError(f"unsupported benchmark schema {schema_version!r}: {path}")


def build_report(paths: tuple[Path, ...]) -> BenchmarkReport:
    """Build summaries and valid baseline/optimized comparisons."""

    if not paths:
        raise ValueError("at least one benchmark artifact is required")

    if len(set(paths)) != len(paths):
        raise ValueError("benchmark artifact paths must not contain duplicates")

    artifacts = tuple(load_artifact(path) for path in paths)
    inference_artifacts = tuple(
        artifact
        for artifact in artifacts
        if artifact.schema_version == BENCHMARK_SCHEMA_VERSION
    )
    comparisons: list[WorkloadComparison] = []
    report_warnings: list[str] = []

    baselines = tuple(
        artifact
        for artifact in inference_artifacts
        if artifact.config.get("backend") == "torch"
    )
    optimized = tuple(
        artifact
        for artifact in inference_artifacts
        if artifact.config.get("backend") == "triton"
    )

    for baseline in baselines:
        for candidate in optimized:
            comparisons.extend(_compare_artifacts(baseline, candidate))

    if baselines and optimized and not comparisons:
        report_warnings.append(
            "Baseline and optimized artifacts shared no workload names."
        )

    if not comparisons:
        report_warnings.append(
            "No baseline-versus-optimized inference comparison was produced."
        )

    if any(artifact.schema_version == SERVING_SCHEMA_VERSION for artifact in artifacts):
        report_warnings.append(
            "Serving artifacts are summarized as deployment-level evidence; "
            "this report does not reinterpret concurrency differences as "
            "isolated batching speedups."
        )

    return BenchmarkReport(
        artifacts=artifacts,
        comparisons=tuple(comparisons),
        warnings=tuple(report_warnings),
    )


def render_markdown(report: BenchmarkReport) -> str:
    """Render a transparent human-readable report."""

    lines = [
        "# TokServe Benchmark Report",
        "",
        "> Generated from benchmark JSON artifacts. Synthetic test fixtures are "
        "not real measurements.",
        "",
        "## Input artifacts",
        "",
    ]

    for artifact in report.artifacts:
        lines.extend(
            (
                f"- `{artifact.path}`",
                f"  - Schema: `{artifact.schema_version}`",
                f"  - Source revision: `{artifact.source_revision}`",
            )
        )

        for warning in artifact.warnings:
            lines.append(f"  - Warning: {warning}")

    lines.extend(("", "## Comparisons", ""))

    if not report.comparisons:
        lines.append("No baseline/optimized comparisons were available.")
    else:
        for comparison in report.comparisons:
            lines.extend(
                (
                    f"### {comparison.workload}",
                    "",
                    f"- Comparable: `{comparison.comparable}`",
                    "- Baseline median latency: "
                    f"`{comparison.baseline_latency.median:.9f} s`",
                    "- Optimized median latency: "
                    f"`{comparison.optimized_latency.median:.9f} s`",
                    "- Latency ratio "
                    "(baseline median / optimized median): "
                    f"`{_format_ratio(comparison.latency_speedup)}`",
                    "- Throughput ratio "
                    "(optimized / baseline): "
                    f"`{_format_ratio(comparison.throughput_speedup)}`",
                    f"- Baseline samples: `{comparison.baseline_latency.sample_count}`",
                    "- Optimized samples: "
                    f"`{comparison.optimized_latency.sample_count}`",
                )
            )

            for warning in comparison.warnings:
                lines.append(f"- Warning: {warning}")

            lines.append("")

    lines.extend(("## Resume-metrics policy", ""))

    valid = tuple(
        comparison
        for comparison in report.comparisons
        if comparison.comparable and comparison.latency_speedup is not None
    )

    if not valid:
        lines.append(
            "No performance ratio in this report is currently eligible for a "
            "resume claim."
        )
    else:
        lines.append(
            "The following ratios are mechanically supported, but any resume "
            "claim must retain the recorded workload, model, dtype, device, "
            "hardware, and synthetic-versus-pretrained qualifiers:"
        )
        lines.append("")

        for comparison in valid:
            lines.append(f"- {_resume_metric(comparison)}")

    if report.warnings:
        lines.extend(("", "## Report warnings", ""))

        for warning in report.warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines).rstrip() + "\n"


def write_report(
    report: BenchmarkReport,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write Markdown or JSON without replacing an artifact accidentally."""

    if output_path.suffix.lower() not in (".md", ".json"):
        raise ValueError("report output must end in .md or .json")

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"report output already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".json":
        contents = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    else:
        contents = render_markdown(report)

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(contents, encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


def _load_inference_artifact(
    path: Path,
    payload: JSONMapping,
) -> ArtifactSummary:
    config = _mapping(payload.get("config"), "config")
    environment = _mapping(payload.get("environment"), "environment")
    source_revision = _string(config, "source_revision")
    raw_measurements = _sequence(payload, "measurements")
    measurements: list[JSONMapping] = []
    warnings: list[str] = []

    if not raw_measurements:
        warnings.append("Artifact contains no measurements.")

    for index, value in enumerate(raw_measurements):
        measurement = _mapping(value, f"measurements[{index}]")
        timing = _mapping(measurement.get("timing"), "timing")
        durations = _float_tuple(timing, "durations_seconds")
        processed_tokens = _integer_tuple(timing, "processed_tokens")

        if len(durations) != len(processed_tokens):
            raise ValueError(f"{path}: duration and token sample counts differ")

        summary = summarize_distribution(durations)
        total_seconds = sum(durations)
        total_tokens = sum(processed_tokens)
        aggregate_throughput = (
            total_tokens / total_seconds if total_seconds > 0.0 else None
        )

        measurements.append(
            {
                "workload": _string(measurement, "workload"),
                "implementation": _string(measurement, "implementation"),
                "timing_scope": _string(measurement, "timing_scope"),
                "throughput_numerator": _string(
                    measurement,
                    "throughput_numerator",
                ),
                "latency_seconds": summary.to_dict(),
                "aggregate_tokens_per_second": aggregate_throughput,
                "peak_allocated_bytes": timing.get("peak_allocated_bytes"),
                "peak_reserved_bytes": timing.get("peak_reserved_bytes"),
            }
        )

        if len(durations) < 20:
            warnings.append(f"{measurement['workload']} has fewer than 20 samples.")

    return ArtifactSummary(
        path=path,
        schema_version=BENCHMARK_SCHEMA_VERSION,
        source_revision=source_revision,
        environment=environment,
        config=config,
        measurements=tuple(measurements),
        warnings=tuple(warnings),
    )


def _load_serving_artifact(
    path: Path,
    payload: JSONMapping,
) -> ArtifactSummary:
    config = _mapping(payload.get("config"), "config")
    source_revision = _string(config, "source_revision")
    client_environment = _mapping(
        payload.get("client_environment"),
        "client_environment",
    )
    load_test = _mapping(payload.get("load_test"), "load_test")
    summary = _mapping(load_test.get("summary"), "load_test.summary")
    requests = _integer(config, "requests")
    warnings = tuple(_string_list(payload, "warnings"))

    measurement: JSONMapping = {
        "workload": "serving",
        "implementation": "tokserve-http",
        "timing_scope": "client_observed_http",
        "mode": _string(config, "mode"),
        "concurrency": _integer(config, "concurrency"),
        "sample_count": requests,
        "summary": summary,
        "pressure": _mapping(payload.get("pressure"), "pressure"),
        "server_config": _mapping(
            payload.get("server_config"),
            "server_config",
        ),
    }

    serving_warnings = list(warnings)

    if requests < MINIMUM_P99_SAMPLES:
        serving_warnings.append(
            "p99 is not eligible because fewer than 100 requests were measured."
        )

    return ArtifactSummary(
        path=path,
        schema_version=SERVING_SCHEMA_VERSION,
        source_revision=source_revision,
        environment=client_environment,
        config=config,
        measurements=(measurement,),
        warnings=tuple(serving_warnings),
    )


def _compare_artifacts(
    baseline: ArtifactSummary,
    optimized: ArtifactSummary,
) -> list[WorkloadComparison]:
    comparability_warnings = _comparability_warnings(
        baseline,
        optimized,
    )
    comparable = not comparability_warnings
    comparisons: list[WorkloadComparison] = []

    baseline_by_workload = {
        _string(measurement, "workload"): measurement
        for measurement in baseline.measurements
    }
    optimized_by_workload = {
        _string(measurement, "workload"): measurement
        for measurement in optimized.measurements
    }

    for workload in sorted(baseline_by_workload.keys() & optimized_by_workload.keys()):
        baseline_measurement = baseline_by_workload[workload]
        optimized_measurement = optimized_by_workload[workload]
        baseline_latency = _distribution_from_mapping(
            baseline_measurement,
            "latency_seconds",
        )
        optimized_latency = _distribution_from_mapping(
            optimized_measurement,
            "latency_seconds",
        )
        baseline_throughput = _optional_float(
            baseline_measurement,
            "aggregate_tokens_per_second",
        )
        optimized_throughput = _optional_float(
            optimized_measurement,
            "aggregate_tokens_per_second",
        )
        warnings = list(comparability_warnings)

        if baseline_latency.median <= 0.0:
            warnings.append("Baseline median latency is zero.")
        if optimized_latency.median <= 0.0:
            warnings.append("Optimized median latency is zero.")
        if baseline_throughput is not None and baseline_throughput <= 0.0:
            warnings.append("Baseline throughput is zero.")
        if optimized_throughput is not None and optimized_throughput <= 0.0:
            warnings.append("Optimized throughput is zero.")

        ratio_allowed = comparable and not warnings
        latency_speedup = (
            baseline_latency.median / optimized_latency.median
            if ratio_allowed
            else None
        )
        throughput_speedup = (
            optimized_throughput / baseline_throughput
            if ratio_allowed
            and baseline_throughput is not None
            and optimized_throughput is not None
            else None
        )

        comparisons.append(
            WorkloadComparison(
                workload=workload,
                comparable=comparable,
                baseline_path=baseline.path,
                optimized_path=optimized.path,
                baseline_latency=baseline_latency,
                optimized_latency=optimized_latency,
                baseline_tokens_per_second=baseline_throughput,
                optimized_tokens_per_second=optimized_throughput,
                latency_speedup=latency_speedup,
                throughput_speedup=throughput_speedup,
                warnings=tuple(warnings),
            )
        )

    return comparisons


def _comparability_warnings(
    baseline: ArtifactSummary,
    optimized: ArtifactSummary,
) -> tuple[str, ...]:
    warnings: list[str] = []

    if baseline.source_revision != optimized.source_revision:
        warnings.append("Source revisions differ.")

    ignored = {"name", "backend", "output_path"}
    baseline_config = {
        key: value for key, value in baseline.config.items() if key not in ignored
    }
    optimized_config = {
        key: value for key, value in optimized.config.items() if key not in ignored
    }

    if baseline_config != optimized_config:
        warnings.append(
            "Benchmark configurations differ beyond backend and output path."
        )

    environment_fields = (
        "operating_system",
        "machine",
        "processor",
        "torch_version",
        "torch_cuda_version",
        "requested_device",
        "resolved_device",
        "cuda_device_name",
        "cuda_capability",
    )

    for field in environment_fields:
        if baseline.environment.get(field) != optimized.environment.get(field):
            warnings.append(f"Environment field differs: {field}.")

    return tuple(warnings)


def _distribution_from_mapping(
    mapping: JSONMapping,
    key: str,
) -> DistributionSummary:
    value = _mapping(mapping.get(key), key)
    p99_value = value.get("p99")

    return DistributionSummary(
        sample_count=_integer(value, "sample_count"),
        minimum=_float(value, "minimum"),
        median=_float(value, "median"),
        p95=_float(value, "p95"),
        p99=(None if p99_value is None else _number(p99_value, "p99")),
        maximum=_float(value, "maximum"),
        mean=_float(value, "mean"),
        coefficient_of_variation=_optional_float(
            value,
            "coefficient_of_variation",
        ),
    )


def _resume_metric(comparison: WorkloadComparison) -> str:
    ratio = comparison.latency_speedup

    if ratio is None:
        raise ValueError("resume metric requires a latency ratio")

    return (
        f"{comparison.workload}: {ratio:.3f}x median-latency ratio "
        "(TokServe PyTorch baseline / TokServe Triton), subject to the "
        "model, dtype, device, hardware, and workload metadata in the "
        "source artifacts."
    )


def _format_ratio(value: float | None) -> str:
    return "not reported" if value is None else f"{value:.3f}x"


def _mapping(value: object, name: str) -> JSONMapping:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")

    return cast(JSONMapping, value)


def _sequence(mapping: JSONMapping, key: str) -> list[object]:
    value = mapping.get(key)

    if not isinstance(value, list):
        raise ValueError(f"{key} must be a JSON array")

    return cast(list[object], value)


def _string(mapping: JSONMapping, key: str) -> str:
    value = mapping.get(key)

    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a nonempty string")

    return value


def _integer(mapping: JSONMapping, key: str) -> int:
    value = mapping.get(key)

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")

    return value


def _float(mapping: JSONMapping, key: str) -> float:
    return _number(mapping.get(key), key)


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")

    return result


def _optional_float(
    mapping: JSONMapping,
    key: str,
) -> float | None:
    value = mapping.get(key)
    return None if value is None else _number(value, key)


def _float_tuple(
    mapping: JSONMapping,
    key: str,
) -> tuple[float, ...]:
    return tuple(_number(value, f"{key}[]") for value in _sequence(mapping, key))


def _integer_tuple(
    mapping: JSONMapping,
    key: str,
) -> tuple[int, ...]:
    values: list[int] = []

    for value in _sequence(mapping, key):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{key} values must be integers")
        values.append(value)

    return tuple(values)


def _string_list(
    mapping: JSONMapping,
    key: str,
) -> list[str]:
    values = _sequence(mapping, key)

    if any(not isinstance(value, str) for value in values):
        raise ValueError(f"{key} values must be strings")

    return cast(list[str], values)
