# flide-mac: Apple Silicon support

This repository is a fork of [OmniJev/Flide](https://github.com/OmniJev/Flide) that runs on **Apple Silicon (MPS)** as well as NVIDIA CUDA.

## What works on Mac

- `flide serve --device auto` (or `--device mps`) on M1 and newer
- Default dtype `float16` on MPS (`bfloat16` remains the CUDA default)
- Qwen3.5 / OneJev-style packed forwards on MPS via **Metal** kernels (`flide/kernels/metal/`) for Gated DeltaNet and causal conv, with vectorized SDPA for packed attention
- Packed `cu_seqlens` GDN/conv run as **one Metal launch** across all segments (`gdn_recurrent_varlen`, `causal_conv1d_varlen_f16`)
- Eager PyTorch fallbacks when Metal is unavailable
- Gemma 4 text path with CUDA graphs disabled
- Images/video through eager vision helpers (slower than CUDA)

## What does not match upstream CUDA

- No Triton / flash-linear-attention on Darwin (optional CUDA extra on Linux)
- No CUDA graphs, FP8, NCCL tensor parallel, or shipped CUDA cubins
- Latency is still below NVIDIA CUDA-graph numbers; Metal covers the GDN/conv hot path today (`torch.mps.compile_shader`), with more kernels (pointwise, custom attn) to follow

## Install (Apple Silicon)

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e .
uv pip install torchvision
flide serve --model OmniJev/OneJev1.6-0.8B --device auto --host 127.0.0.1
```

Use a small checkpoint first. Large 27B models need plenty of unified memory; lower `--session-cache-gb` and prefer `--device mps` with `float16`.

## Linux / NVIDIA

Behavior stays close to upstream: install with the `cuda` extra if needed so `flash-linear-attention` is present.

```bash
uv pip install -e ".[cuda]"
```

## Upstream

Engine changes that affect CUDA speed or answers are not proposed upstream until Mac parity tests are stable. See `CONTRIBUTING.md` in the original project.
