# Milestone 10 Profiling and Optimization Results

## Environment

- GPU: Tesla T4
- PyTorch: 2.11.0+cu128
- PyTorch CUDA build: 12.8
- Triton: 3.6.0
- Backend: Triton
- Data type: float32

## Workload

The end-to-end profiling workload used:

- Batch size: 1
- Prompt length: 128
- Hidden size: 128
- Query heads: 8
- Key/value heads: 2
- Head dimension: 16
- Decoder layers: 2

## Baseline

Repeated baseline measurements produced approximately:

- Prefill: 2.50–2.78 ms
- Decode: 2.47–2.50 ms after excluding the initial warmup outlier

The PyTorch profiler showed four `aten::any` calls for a two-layer model.

## Bottleneck

The Triton RoPE wrapper checked whether CUDA position values were negative
before calling `rotary_cos_sin`.

`rotary_cos_sin` performed the same validation again. This caused two
negative-position reductions per decoder layer and could introduce redundant
CUDA-to-host synchronization.

## Optimization

The duplicate negative-position check was removed from
`triton_apply_rope`.

The validation inside `rotary_cos_sin` remains in place, so negative
positions are still rejected and the public behavior is unchanged.

## Profiler result

After the optimization, prefill showed two `aten::any` calls instead of
four.

The decode profile also showed reductions in associated operations:

- `aten::copy_`: 16 calls to 14 calls
- Device-to-device copies: 6 calls to 4 calls

This confirms that the intended redundant work was removed.

## End-to-end results

Three post-optimization measurements produced:

| Workload | Run 1 | Run 2 | Run 3 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 2.526544 ms | 2.575056 ms | 2.476496 ms | 2.526032 ms |
| Decode | 2.398768 ms | 2.417360 ms | 2.421136 ms | 2.412421 ms |

Compared with the steady baseline averages:

- Prefill improved from approximately 2.612 ms to 2.526 ms, or about 3.3%.
- Decode improved from approximately 2.484 ms to 2.412 ms, or about 2.9%.

These percentages describe the measured Tesla T4 runs and are not universal
performance guarantees.

## RoPE kernel audit

After the optimization, Triton was faster than the PyTorch reference in
seven of eight measured RoPE cases.

The only exception was float32 one-token decode:

- PyTorch: 0.345568 ms
- Triton: 0.350480 ms

This difference was approximately 1.4%.

## Correctness and regression checks

The optimization retained the negative-position validation inside
`rotary_cos_sin`.

Formatting, linting, strict type checking, local tests, targeted CUDA tests,
and the complete Colab test suite passed. Expected skips were limited to
unsupported platform features such as MPS and native CUDA bfloat16.

## Conclusion

The optimization removed measured redundant GPU work without changing the
public API or correctness behavior. It produced a small, repeatable
improvement in the profiled end-to-end workload and is retained.