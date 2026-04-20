if __name__ == "__main__":
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        "/data/models/Qwen/Qwen3-4B/",
        enforce_eager=False,
        tensor_parallel_size=1,
        chunked_prefill=True,
        gpu_memory_utilization=0.4,
        profiler_dir="",
        speculative_draft_model="/data/models/Qwen/Qwen3-0.6B",
        speculative_num_draft_tokens=3,
    )
    # llm.start_profile()
    sampling_params = SamplingParams(temperature=0, max_tokens=128)
    prompts = ["世界上最长的河流是哪一条？", "地球的赤道有多长？"]
    outputs = llm.generate(prompts, sampling_params)
    # llm.stop_profile()

    print(f"response: {outputs[0]['text']} \n response: {outputs[1]['text']}")
