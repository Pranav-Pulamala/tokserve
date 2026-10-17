"""Create a validated report from TokServe benchmark JSON artifacts."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from tokserve.benchmarking.report import build_report, write_report


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark-report CLI."""

    parser = argparse.ArgumentParser(
        description="Summarize and compare TokServe benchmark artifacts.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        dest="inputs",
        help="benchmark JSON artifact; repeat for multiple artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new .md or .json report path",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing report explicitly",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    """Load artifacts and write one report."""

    parsed = build_parser().parse_args(arguments)
    paths = tuple(parsed.inputs)
    output_path: Path = parsed.output

    report = build_report(paths)
    written_path = write_report(
        report,
        output_path,
        overwrite=bool(parsed.overwrite),
    )

    print(f"wrote benchmark report: {written_path}")
    print(f"input artifacts: {len(report.artifacts)}")
    print(f"workload comparisons: {len(report.comparisons)}")

    comparable = sum(comparison.comparable for comparison in report.comparisons)
    print(f"comparable workload pairs: {comparable}")

    if report.warnings:
        print("report warnings:")

        for warning in report.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
