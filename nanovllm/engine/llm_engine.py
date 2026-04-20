import atexit
import math
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (q / 100.0) * (len(sorted_values) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return sorted_values[low]
    weight = rank - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self._last_generate_metrics = {}
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def step(self):
        seqs = self.scheduler.schedule()
        token_ids, seq_need_compute_logits = self.model_runner.call("run", seqs)
        self.scheduler.postprocess(seqs, token_ids, seq_need_compute_logits)
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in seqs if seq.is_finished)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def start_profile(self):
        self.model_runner.call("profile", True)

    def stop_profile(self):
        self.model_runner.call("profile", False)

    def get_last_generate_metrics(self):
        return dict(self._last_generate_metrics)

    def reset_last_generate_metrics(self):
        self._last_generate_metrics = {}

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        collect_metrics: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        tracked_seqs = []
        for prompt, sp in zip(prompts, sampling_params):
            seq = self.add_request(prompt, sp)
            if collect_metrics:
                tracked_seqs.append(seq)
        outputs = {}
        num_total_tokens = 0
        t = perf_counter()
        if collect_metrics:
            first_token_ts = {seq.seq_id: None for seq in tracked_seqs}
            finish_ts = {seq.seq_id: None for seq in tracked_seqs}
            output_tokens = {seq.seq_id: 0 for seq in tracked_seqs}
        while not self.is_finished():
            if collect_metrics:
                prev_completion_tokens = {
                    seq.seq_id: seq.num_completion_tokens for seq in tracked_seqs
                }
            output, num_step_tokens = self.step()
            num_total_tokens += num_step_tokens
            if use_tqdm:
                total_throughput = num_total_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "total_throughput": f"{int(total_throughput)}tok/s",
                    }
                )
            if collect_metrics:
                now = perf_counter()
                for seq in tracked_seqs:
                    current_completion_tokens = seq.num_completion_tokens
                    delta = (
                        current_completion_tokens - prev_completion_tokens[seq.seq_id]
                    )
                    if delta > 0 and first_token_ts[seq.seq_id] is None:
                        first_token_ts[seq.seq_id] = now
                    output_tokens[seq.seq_id] = current_completion_tokens
                    if seq.is_finished and finish_ts[seq.seq_id] is None:
                        finish_ts[seq.seq_id] = now

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        if collect_metrics:
            total_time = perf_counter() - t
            total_output_tokens = sum(output_tokens.values())
            output_throughput = (
                total_output_tokens / total_time if total_time > 0 else 0.0
            )
            ttft_ms = [
                (ts - t) * 1000.0 for ts in first_token_ts.values() if ts is not None
            ]
            tpop_ms = []
            for seq in tracked_seqs:
                num_output_tokens = output_tokens[seq.seq_id]
                seq_first_token_ts = first_token_ts[seq.seq_id]
                seq_finish_ts = finish_ts[seq.seq_id]
                if (
                    num_output_tokens <= 0
                    or seq_first_token_ts is None
                    or seq_finish_ts is None
                ):
                    continue
                if num_output_tokens == 1:
                    per_output_token = seq_finish_ts - t
                else:
                    per_output_token = (seq_finish_ts - seq_first_token_ts) / (
                        num_output_tokens - 1
                    )
                tpop_ms.append(per_output_token * 1000.0)
            self._last_generate_metrics = {
                "num_requests": len(tracked_seqs),
                "total_output_tokens": total_output_tokens,
                "total_time_s": total_time,
                "output_throughput_tok_s": output_throughput,
                "mean_ttft_ms": _mean(ttft_ms),
                "p99_ttft_ms": _percentile(ttft_ms, 99),
                "mean_tpop_ms": _mean(tpop_ms),
                "p99_tpop_ms": _percentile(tpop_ms, 99),
            }
        else:
            self._last_generate_metrics = {}
        if use_tqdm:
            pbar.close()
        return outputs
