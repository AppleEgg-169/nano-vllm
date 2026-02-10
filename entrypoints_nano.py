if __name__ == "__main__":
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        "/data/models/Qwen/Qwen3-8B",
        enforce_eager=True,
        tensor_parallel_size=1,
        chunked_prefill=True,
    )
    sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
    prompts = ["世界上最长的河流是哪一条？", "地球的赤道有多长？"]
    outputs = llm.generate(prompts, sampling_params)

    print(f"response: {outputs[0]['text']} \n response: {outputs[1]['text']}")
