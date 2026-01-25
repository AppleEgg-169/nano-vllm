# Nano-vLLM-v1

A lightweight [vLLM](https://github.com/vllm-project/vllm) implementation built from scratch. While built upon the foundation of [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), this project significantly re-engineers the core architecture to **reproduce the [vLLM v1 scheduler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)** and introduces **Chunked Prefill**.

## Key Features

* 🚀 **Fast offline inference** - Comparable online inference speeds and offline throughput performance to vLLM v1.
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Paged Attention, Prefix Caching, Chunked Prefill, Tensor Parallelism, Torch Compilation, and full reproduction of the vLLM v1 scheduling strategy, etc.

## Quick Start

See `example.py` (offline) and `serving_bench.py` (online) for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:

offline usage example:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

online benchmarking:
```
python serving_bench.py \
--model /path/to/Qwen3-14B/ \
--request-rate 10 \
--num-requests 1024 \
--tensor-parallel-size 1 \
--max-num-batched-tokens 1024 \
--max-num-seqs 1024 \
--random-input-len 128 \
--random-output-len 100 \
--chunked-prefill \
--enforce-eager
```
