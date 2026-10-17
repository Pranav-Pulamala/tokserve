# TokServe Reproducible Benchmark Runbook

This runbook generates reproducible TokServe benchmark evidence. It does not
contain prepopulated performance claims.

## Evidence policy

- CPU tiny-model runs are smoke tests, not GPU performance evidence.
- Direct PyTorch/Triton ratios require the same source revision, model,
  weights, prompt token IDs, dtype, device, hardware, output length,
  sampling mode, cache configuration, warmup count, and repetition count.
- Latency speedup means baseline latency divided by optimized latency.
- Throughput speedup means optimized throughput divided by baseline throughput.
- Serving sequential/concurrent results are deployment-level comparisons,
  not isolated tensor-batching speedups.
- Results with correctness or parity failures must not be used.
- Do not report p99 with fewer than 100 observations.
- Do not commit downloaded weights, large traces, secrets, private prompts,
  or machine-specific caches.
- Small result artifacts may be committed only after manual inspection.

## 1. Verify the source and environment

Run from the TokServe project root:

```bash
git status
git rev-parse HEAD
./.venv/bin/python3 --version
./.venv/bin/python3 -c "import torch; print('torch:', torch.__version__); print('cuda build:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

The working tree should be clean before collecting final evidence. Save the
exact commit printed by `git rev-parse HEAD`.

In later commands, replace `SOURCE_REVISION` with that complete commit SHA.

## 2. Run the CPU-safe smoke benchmark

```bash
./.venv/bin/python3 -m benchmarks.inference_baseline \
  --source-revision SOURCE_REVISION \
  --output /tmp/tokserve-cpu-smoke.json \
  --device cpu \
  --dtype float32 \
  --prompt-token-id 1 \
  --prompt-token-id 2 \
  --prompt-token-id 3 \
  --prompt-token-id 4 \
  --max-output-tokens 2 \
  --warmup 1 \
  --repetitions 2 \
  --seed 1510 \
  --batch-size 1 \
  --vocab-size 32 \
  --hidden-size 16 \
  --intermediate-size 32 \
  --num-hidden-layers 1 \
  --num-attention-heads 4 \
  --num-key-value-heads 2 \
  --max-position-embeddings 32
```

Inspect the artifact:

```bash
./.venv/bin/python3 -m json.tool /tmp/tokserve-cpu-smoke.json
```

Classification: deterministic synthetic-model CPU smoke test. It is not
evidence of GPU speedup or pretrained-model performance.

## 3. Prepare the NVIDIA GPU environment

Use Colab or another NVIDIA CUDA environment. Clone or pull the repository,
check out the intended commit, and install the project dependencies.

Verify the environment:

```bash
git status
git rev-parse HEAD
python3 -c "import torch, triton, tokserve; print('torch:', torch.__version__); print('cuda build:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0)); print('triton:', triton.__version__); print('tokserve:', tokserve.__file__)"
```

Every artifact in a direct comparison must use this same source SHA.

## 4. Run the matched PyTorch baseline

```bash
python3 -m benchmarks.inference_baseline \
  --source-revision SOURCE_REVISION \
  --output /tmp/tokserve-baseline-gpu.json \
  --device cuda \
  --dtype float32 \
  --prompt-token-id 1 \
  --prompt-token-id 2 \
  --prompt-token-id 3 \
  --prompt-token-id 4 \
  --max-output-tokens 8 \
  --warmup 5 \
  --repetitions 20 \
  --seed 1510 \
  --batch-size 1 \
  --vocab-size 256 \
  --hidden-size 128 \
  --intermediate-size 256 \
  --num-hidden-layers 2 \
  --num-attention-heads 8 \
  --num-key-value-heads 2 \
  --max-position-embeddings 256
```

## 5. Run the matched Triton benchmark

```bash
python3 -m benchmarks.inference_optimized \
  --source-revision SOURCE_REVISION \
  --output /tmp/tokserve-optimized-gpu.json \
  --prompt-token-id 1 \
  --prompt-token-id 2 \
  --prompt-token-id 3 \
  --prompt-token-id 4 \
  --max-output-tokens 8 \
  --warmup 5 \
  --repetitions 20 \
  --seed 1510 \
  --batch-size 1 \
  --vocab-size 256 \
  --hidden-size 128 \
  --intermediate-size 256 \
  --num-hidden-layers 2 \
  --num-attention-heads 8 \
  --num-key-value-heads 2 \
  --max-position-embeddings 256
```

The optimized benchmark must pass its numerical-parity checks before its
timings are considered valid.

## 6. Run the cache experiments

Uncached:

```bash
python3 -m benchmarks.cache_memory \
  --source-revision SOURCE_REVISION \
  --output /tmp/tokserve-cache-none.json \
  --cache-mode none \
  --backend torch \
  --device cuda \
  --dtype float32 \
  --max-output-tokens 8 \
  --warmup 5 \
  --repetitions 20 \
  --seed 1510
```

Contiguous cache:

```bash
python3 -m benchmarks.cache_memory \
  --source-revision SOURCE_REVISION \
  --output /tmp/tokserve-cache-contiguous.json \
  --cache-mode contiguous \
  --backend torch \
  --device cuda \
  --dtype float32 \
  --max-output-tokens 8 \
  --warmup 5 \
  --repetitions 20 \
  --seed 1510
```

Paged cache:

```bash
python3 -m benchmarks.cache_memory \
  --source-revision SOURCE_REVISION \
  --output /tmp/tokserve-cache-paged.json \
  --cache-mode paged \
  --backend torch \
  --device cuda \
  --dtype float32 \
  --max-output-tokens 8 \
  --warmup 5 \
  --repetitions 20 \
  --seed 1510
```

Theoretical cache capacity and measured CUDA peak allocation/reservation are
different metrics and must remain separately labeled.

## 7. Run sequential serving

Start a clean server in one terminal:

```bash
TOKSERVE_BENCHMARK_DEVICE=cuda \
TOKSERVE_BENCHMARK_BACKEND=triton \
TOKSERVE_BENCHMARK_SEED=1510 \
python3 -m uvicorn tokserve.api.benchmark_app:app \
  --host 127.0.0.1 \
  --port 8000
```

In another terminal:

```bash
python3 -m benchmarks.serving_benchmark \
  --source-revision SOURCE_REVISION \
  --base-url http://127.0.0.1:8000 \
  --output /tmp/tokserve-serving-sequential.json \
  --mode sequential \
  --requests 100 \
  --concurrency 1 \
  --prompt "1 2 3 4" \
  --prompt "5 6" \
  --prompt "7 8 9" \
  --prompt "10" \
  --max-new-tokens 8 \
  --warmup-requests 5 \
  --timeout-seconds 120
```

Stop the server afterward. Server metrics are cumulative, so the concurrent
test requires a clean restart.

## 8. Run concurrent serving

Start the same server again:

```bash
TOKSERVE_BENCHMARK_DEVICE=cuda \
TOKSERVE_BENCHMARK_BACKEND=triton \
TOKSERVE_BENCHMARK_SEED=1510 \
python3 -m uvicorn tokserve.api.benchmark_app:app \
  --host 127.0.0.1 \
  --port 8000
```

In another terminal:

```bash
python3 -m benchmarks.serving_benchmark \
  --source-revision SOURCE_REVISION \
  --base-url http://127.0.0.1:8000 \
  --output /tmp/tokserve-serving-concurrent.json \
  --mode concurrent \
  --requests 100 \
  --concurrency 8 \
  --prompt "1 2 3 4" \
  --prompt "5 6" \
  --prompt "7 8 9" \
  --prompt "10" \
  --max-new-tokens 8 \
  --warmup-requests 5 \
  --timeout-seconds 120
```

Stop the server afterward.

The concurrent artifact must demonstrate overlapping requests through actual
batching metrics. The sequential/concurrent comparison remains a
deployment-level comparison.

## 9. Generate the report

```bash
python3 -m benchmarks.summarize_results \
  --input /tmp/tokserve-baseline-gpu.json \
  --input /tmp/tokserve-optimized-gpu.json \
  --input /tmp/tokserve-cache-none.json \
  --input /tmp/tokserve-cache-contiguous.json \
  --input /tmp/tokserve-cache-paged.json \
  --input /tmp/tokserve-serving-sequential.json \
  --input /tmp/tokserve-serving-concurrent.json \
  --output /tmp/tokserve-benchmark-report.md
```

Inspect it:

```bash
sed -n '1,280p' /tmp/tokserve-benchmark-report.md
```

## 10. Evaluate claim eligibility

A measurement is potentially suitable for a performance claim only if:

- correctness and numerical parity passed;
- both sides used the same source revision and environment;
- configurations are equivalent except for the execution backend;
- the generated report marks the comparison comparable;
- the measurement came from the intended GPU;
- sample counts, units, workload, and hardware are stated;
- no failures were hidden;
- synthetic-model status is disclosed;
- regressions and neutral results are reported honestly.

Do not claim superiority over vLLM, Hugging Face, or FlashAttention. This
suite does not implement validated comparisons against those systems.