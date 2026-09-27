"""CPU-safe tests for profiler configuration and selection."""

from pathlib import Path

import pytest

from benchmarks.profile_model import (
    selected_workloads,
    trace_path_for,
)
from tokserve.profiling.torch_profiler import ProfilerSettings


def test_profiler_settings_defaults_are_bounded() -> None:
    settings = ProfilerSettings()

    assert settings.warmup_iterations == 5
    assert settings.row_limit == 20
    assert settings.record_shapes
    assert settings.profile_memory


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("prefill", ("prefill",)),
        ("decode", ("decode",)),
        ("both", ("prefill", "decode")),
    ],
)
def test_workload_selection(
    requested: str,
    expected: tuple[str, ...],
) -> None:
    assert selected_workloads(requested) == expected


def test_trace_path_is_optional() -> None:
    assert trace_path_for("prefill", enabled=False) is None
    assert trace_path_for(
        "decode",
        enabled=True,
    ) == Path("profiles/decode.json")


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("warmup_iterations", -1, "warmup_iterations"),
        ("row_limit", 0, "row_limit"),
    ],
)
def test_invalid_profiler_settings_are_rejected(
    keyword: str,
    value: int,
    message: str,
) -> None:
    arguments = {keyword: value}

    with pytest.raises(ValueError, match=message):
        ProfilerSettings(**arguments)
