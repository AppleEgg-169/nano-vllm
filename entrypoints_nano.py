if __name__ == "__main__":
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        "/data/models/Qwen/Qwen3-0.6B", enforce_eager=True, tensor_parallel_size=1
    )
    sampling_params = SamplingParams(temperature=1, max_tokens=128)
    prompts = ["What is the longest river in the world."]
    outputs = llm.generate(prompts, sampling_params)

    print(f"response: {outputs[0]['text']}")
