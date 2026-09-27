"""CPU-safe tests for kernel-audit selection."""

import pytest

from benchmarks.kernel_audit import selected_operations


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("rmsnorm", ("rmsnorm",)),
        ("rope", ("rope",)),
        ("attention", ("attention",)),
        ("all", ("rmsnorm", "rope", "attention")),
    ],
)
def test_selected_operations(
    requested: str,
    expected: tuple[str, ...],
) -> None:
    assert selected_operations(requested) == expected


def test_unknown_operation_is_rejected() -> None:
    with pytest.raises(ValueError, match="operation must be"):
        selected_operations("unknown")
