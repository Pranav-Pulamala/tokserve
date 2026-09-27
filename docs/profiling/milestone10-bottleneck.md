# Milestone 10 Bottleneck Analysis

## Environment

- GPU: Tesla T4
- PyTorch: 2.11.0+cu128
- PyTorch CUDA build: 12.8
- Triton: 3.6.0

## End-to-end baseline

Repeated CUDA-event measurements produced approximately:

- Prefill: 2.50–2.78 ms
- Decode: 2.47–2.50 ms after the initial warmup outlier

The workload uses batch size 1, a prompt length of 128, hidden size 128,
two decoder layers, eight query heads, and two key/value heads.

## Profiler observations

For prefill, the largest CUDA contributors were:

- Matrix multiplication: 264.285 microseconds across 15 calls
- Tiled attention: 116.382 microseconds across 2 calls
- Tensor copies: 51.486 microseconds across 12 calls
- RMSNorm kernels: 19.039 microseconds across 5 calls
- RoPE kernels: 12.993 microseconds across 4 calls
- `aten::any`: 28.575 microseconds across 4 calls

For decode, the largest CUDA contributors included:

- Tiled attention: 87.230 microseconds across 2 calls
- Matrix multiplication: 81.950 microseconds across 15 calls
- Tensor copies: 55.456 microseconds across 16 calls
- RMSNorm kernels: 16.544 microseconds across 5 calls
- RoPE kernels: 11.520 microseconds across 4 calls
- `aten::arange`: 12.287 microseconds across 10 calls
- `aten::_local_scalar_dense`: 9.984 microseconds across 6 calls

## Kernel audit

RMSNorm was consistently faster with Triton for every measured shape, so
it is not the first optimization target.

Tiled attention was faster for small inputs but slower for several larger
prefill and decode cases. Improving it would require a larger kernel-design
change and is therefore not the smallest low-risk optimization.

RoPE results were mixed. The actual Triton RoPE kernels accounted for only
about 11–13 microseconds in the model profiles, while complete standalone
RoPE calls took roughly 0.4–0.6 milliseconds.

## Selected bottleneck

The first optimization target is redundant validation in the Triton RoPE
call path.

`triton_apply_rope` checks whether any position is negative using
`torch.any(positions < 0)`. It then calls `rotary_cos_sin`, which performs
the same check again.

Because positions are CUDA tensors, evaluating these checks from Python can
introduce device-to-host synchronization. With two decoder layers, the
duplicate validation explains the four observed `aten::any` calls.

## Planned optimization

Remove the duplicate negative-position check from `triton_apply_rope` while
retaining the check inside `rotary_cos_sin`.

This preserves validation behavior because every call still passes through
`rotary_cos_sin`, while reducing redundant CUDA work and synchronization.

## Validation plan

After the change:

1. Verify negative positions are still rejected.
2. Run formatting, linting, type checking, and the complete test suite.
3. Run the GPU RoPE and end-to-end Triton tests in Colab.
4. Re-run the profiler and confirm that `aten::any` calls decrease from four
   to two for the two-layer workload.
5. Re-run the RoPE and end-to-end benchmarks.
6. Keep the optimization only if correctness is preserved and the measured
   results support it.