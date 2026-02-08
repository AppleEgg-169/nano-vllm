from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    def __init__(self, config: Config):
        self.enable_chunked = config.chunked_prefill
        self.max_model_len = config.max_model_len
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        assert len(seq) <= self.max_model_len - 1, (
            "Sequence length exceeds max_model_len"
        )
        self.waiting.append(seq)

    def schedule(self) -> list[Sequence]:
        scheduled_seqs = []
        seq_budget = self.max_num_seqs
        token_budget = self.max_num_batched_tokens
        preempted = False

        # decode
        while self.running and seq_budget > 0 and token_budget > 0:
            seq = self.running.popleft()
            num_new_tokens = seq.num_tokens - seq.num_cached_tokens
            if self.enable_chunked:
                num_new_tokens = min(num_new_tokens, token_budget)
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - seq.num_cached_tokens
            )

            # TODO: num_new_tokens can be 0 when chunked_prefill is enabled!
            assert num_new_tokens > 0
            while not self.block_manager.can_append(seq, num_new_tokens):
                preempted = True
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_new_tokens = num_new_tokens
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
                token_budget -= num_new_tokens
                seq_budget -= 1

        # prefill
        if not preempted:
            while self.waiting and seq_budget > 0 and token_budget > 0:
                seq = self.waiting[0]
                assert not seq.block_table
                num_new_tokens, num_cached_tokens_used, num_cached_tokens_free = (
                    self.block_manager.compute_num_tokens(seq)
                )
                assert num_new_tokens > 0
                if self.enable_chunked:
                    num_new_tokens = min(num_new_tokens, token_budget)

                if token_budget < num_new_tokens or not self.block_manager.can_allocate(
                    num_new_tokens + num_cached_tokens_free
                ):
                    break
                seq.num_new_tokens = num_new_tokens
                seq_budget -= 1
                token_budget -= seq.num_new_tokens
                self.block_manager.allocate(seq)
                assert (
                    seq.num_cached_tokens
                    == num_cached_tokens_used + num_cached_tokens_free
                )
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                scheduled_seqs.append(seq)

        assert scheduled_seqs
        self.running.clear()
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self, seqs: list[Sequence], token_ids: list[int], seq_need_compute_logits
    ) -> list[bool]:
        assert len(token_ids) == len(seq_need_compute_logits)
        for seq_index, token_id in zip(seq_need_compute_logits, token_ids):
            seq = seqs[seq_index]
            seq.append_token(token_id)
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

        for seq in seqs:
            if seq.status != SequenceStatus.FINISHED:
                seq.num_cached_tokens += seq.num_new_tokens
                seq.num_new_tokens = 0
