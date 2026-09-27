"""Validated CUDA wrapper for Triton RoPE."""

import torch

from tokserve.engine.rope import rotary_cos_sin


def triton_apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    *,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the project's half-rotation RoPE to Q and K."""
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must have shape (B, H, T, Dh)")

    if query.shape[0] != key.shape[0]:
        raise ValueError("query and key batch sizes must match")

    if query.shape[-2:] != key.shape[-2:]:
        raise ValueError("query and key sequence and head dimensions must match")

    if query.dtype != key.dtype:
        raise ValueError("query and key must use the same dtype")

    if query.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("query and key must use a supported floating dtype")

    if not query.is_cuda or not key.is_cuda or not positions.is_cuda:
        raise ValueError("query, key, and positions must be CUDA tensors")

    if query.device != key.device or positions.device != query.device:
        raise ValueError("query, key, and positions must use the same device")

    sequence_length = query.shape[-2]
    head_dim = query.shape[-1]

    if positions.shape != (sequence_length,):
        raise ValueError("positions must have shape (T,)")

    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("positions must contain integers")

    if head_dim < 2 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even integer")

    if theta <= 0.0:
        raise ValueError("theta must be positive")

    cosine, sine = rotary_cos_sin(
        positions,
        head_dim,
        theta=theta,
        dtype=query.dtype,
    )

    from tokserve.kernels._triton_rope import launch_rope

    rotated_query: torch.Tensor = launch_rope(query, cosine, sine)
    rotated_key: torch.Tensor = launch_rope(key, cosine, sine)
    return rotated_query, rotated_key
