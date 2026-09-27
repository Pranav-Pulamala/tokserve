"""Profile TokServe prefill and cached decode with torch.profiler."""

import argparse
from pathlib import Path
from typing import Literal, TypeAlias, cast

import torch

from tokserve.profiling.torch_profiler import (
    ProfilerSettings,
    profile_workload,
)
from tokserve.profiling.workloads import (
    InferenceProfilingWorkloads,
    ProfilingConfig,
    WorkloadKind,
    environment_metadata,
)

RequestedWorkload: TypeAlias = Literal["prefill", "decode", "both"]


def parse_args() -> argparse.Namespace:
    """Parse profiling command-line options."""

    parser = argparse.ArgumentParser(
        description="Profile TokServe CUDA inference workloads.",
    )
    parser.add_argument(
        "--workload",
        choices=("prefill", "decode", "both"),
        default="both",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="export local Chrome trace JSON files under profiles/",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )
    return parser.parse_args()


def selected_workloads(
    requested: RequestedWorkload,
) -> tuple[WorkloadKind, ...]:
    """Expand the requested CLI workload selection."""

    if requested == "both":
        return ("prefill", "decode")

    return (requested,)


def trace_path_for(
    kind: WorkloadKind,
    *,
    enabled: bool,
) -> Path | None:
    """Return a local trace path when export is enabled."""

    if not enabled:
        return None

    return Path("profiles") / f"{kind}.json"


@torch.inference_mode()
def main() -> None:
    """Profile deterministic prefill and decode operations."""

    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This profiler requires an NVIDIA CUDA GPU")

    requested = cast(RequestedWorkload, args.workload)
    trace_enabled = bool(args.trace)
    row_limit = int(args.row_limit)
    warmup = int(args.warmup)

    settings = ProfilerSettings(
        warmup_iterations=warmup,
        row_limit=row_limit,
    )
    config = ProfilingConfig()
    workloads = InferenceProfilingWorkloads(
        config,
        device=torch.device("cuda:0"),
    )

    for name, value in environment_metadata().items():
        print(f"{name}: {value}")

    print(
        "configuration: "
        f"backend={config.backend} "
        f"dtype={str(config.dtype).removeprefix('torch.')} "
        f"B={config.batch_size} "
        f"prompt_length={config.prompt_length} "
        f"D={config.hidden_size} "
        f"Hq={config.num_attention_heads} "
        f"Hkv={config.num_key_value_heads} "
        f"Dh={config.head_dim} "
        f"layers={config.num_hidden_layers}"
    )

    for kind in selected_workloads(requested):
        print()
        print(f"===== {kind.upper()} — TOP CUDA OPERATIONS =====")
        summary = profile_workload(
            workloads,
            kind,
            settings=settings,
            trace_path=trace_path_for(
                kind,
                enabled=trace_enabled,
            ),
        )
        print(summary)

        if trace_enabled:
            print(f"trace: profiles/{kind}.json")


if __name__ == "__main__":
    main()
