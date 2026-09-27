"""Run TokServe's existing custom-kernel microbenchmarks."""

import argparse
from typing import Literal, TypeAlias

Operation: TypeAlias = Literal[
    "rmsnorm",
    "rope",
    "attention",
]


def selected_operations(requested: str) -> tuple[Operation, ...]:
    """Return the benchmark operations selected by the CLI."""

    if requested == "all":
        return ("rmsnorm", "rope", "attention")

    if requested == "rmsnorm":
        return ("rmsnorm",)

    if requested == "rope":
        return ("rope",)

    if requested == "attention":
        return ("attention",)

    raise ValueError("operation must be 'rmsnorm', 'rope', 'attention', or 'all'")


def run_operation(operation: Operation) -> None:
    """Lazily import and run one CUDA benchmark."""

    print()
    print("=" * 72)
    print(f"KERNEL AUDIT: {operation.upper()}")
    print("=" * 72)

    if operation == "rmsnorm":
        from benchmarks.rmsnorm import main

        main()
        return

    if operation == "rope":
        from benchmarks.rope import main

        main()
        return

    from benchmarks.attention import main

    main()


def parse_args() -> argparse.Namespace:
    """Parse kernel-audit command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run TokServe custom CUDA kernel benchmarks.",
    )
    parser.add_argument(
        "--operation",
        choices=("rmsnorm", "rope", "attention", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    """Run every selected benchmark."""

    args = parse_args()

    for operation in selected_operations(str(args.operation)):
        run_operation(operation)


if __name__ == "__main__":
    main()
