import os
from itertools import product
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


# NUM_SEQS_CASES = [1, 8, 32, 64]
# MAX_INPUT_LEN_CASES = [128, 512, 1024, 2048]
# MAX_OUTPUT_LEN_CASES = [128, 256, 512]

NUM_SEQS_CASES = [10]
MAX_INPUT_LEN_CASES = [13000]
MAX_OUTPUT_LEN_CASES = [128, 512]


def run_case(
    llm: LLM,
    num_seqs: int,
    max_input_len: int,
    max_output_len: int,
):
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(max_input_len)] for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.06,
            ignore_eos=True,
            max_tokens=max_output_len,
        )
        for _ in range(num_seqs)
    ]

    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]
    llm.generate(
        prompt_token_ids,
        sampling_params,
        use_tqdm=False,
        collect_metrics=True,
    )
    return llm.get_last_generate_metrics()


def main():
    seed(0)

    model_path = os.path.expanduser("/data/models/Qwen/Qwen3-4B/")
    llm = LLM(
        model_path,
        enforce_eager=False,
        gpu_memory_utilization=0.9,
        chunked_prefill=True,
        tensor_parallel_size=1,
    )

    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
    cases = list(product(NUM_SEQS_CASES, MAX_INPUT_LEN_CASES, MAX_OUTPUT_LEN_CASES))
    results = []
    total_cases = len(cases)

    print(f"Total benchmark cases: {total_cases}")
    for idx, (num_seqs, max_input_len, max_output_len) in enumerate(cases, start=1):
        print(
            f"[{idx}/{total_cases}] "
            f"num_seqs={num_seqs}, max_input_len={max_input_len}, max_output_len={max_output_len}"
        )
        try:
            metrics = run_case(llm, num_seqs, max_input_len, max_output_len)
            results.append(
                {
                    "num_seqs": num_seqs,
                    "max_input_len": max_input_len,
                    "max_output_len": max_output_len,
                    **metrics,
                }
            )
            print(
                f"  Throughput={metrics['output_throughput_tok_s']:.2f}tok/s, "
                f"MeanTTFT={metrics['mean_ttft_ms']:.2f}ms, "
                f"P99TTFT={metrics['p99_ttft_ms']:.2f}ms, "
                f"MeanTPOP={metrics['mean_tpop_ms']:.2f}ms, "
                f"P99TPOP={metrics['p99_tpop_ms']:.2f}ms"
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")

    print("\n=== Summary ===")
    print(
        "num_seqs,max_input_len,max_output_len,total_output_tokens,total_time_s,"
        "output_throughput_tok_s,mean_ttft_ms,p99_ttft_ms,mean_tpop_ms,p99_tpop_ms"
    )
    for result in results:
        print(
            f"{result['num_seqs']},"
            f"{result['max_input_len']},"
            f"{result['max_output_len']},"
            f"{result['total_output_tokens']},"
            f"{result['total_time_s']:.4f},"
            f"{result['output_throughput_tok_s']:.2f},"
            f"{result['mean_ttft_ms']:.2f},"
            f"{result['p99_ttft_ms']:.2f},"
            f"{result['mean_tpop_ms']:.2f},"
            f"{result['p99_tpop_ms']:.2f}"
        )


if __name__ == "__main__":
    main()
