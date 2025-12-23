import numpy as np
import torch
import math
import time
from vllm.config import VllmConfig
import torch.cuda.nvtx as nvtx
from ucm.sparse.base import (
    INVALID_SLOT,
    UcmSparseBase,
    UcmSparseMetadata,
    UcmSparseRole,
)
import sys
sys.path.append("/root/hk/cuda")
import esa_interface as esa_lib
esa_retrieval = esa_lib.esa_retrieval
esa_repre = esa_lib.esa_repre
esa_copy = esa_lib.esa_copy
esa_scatter_copy = esa_lib.esa_scatter_copy


class ReprePool:
    def __init__(self, capability):
        self.repre_blocks = list(range(capability-1, -1, -1))

    def allocate(self, num_blocks):
        res = []
        for _ in range(num_blocks):
            res.append(self.repre_blocks.pop())
        return res

    def free(self, blocks):
        for e in blocks:
            self.repre_blocks.append(e)

class ESA(UcmSparseBase):
    # handle batch
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        super().__init__(vllm_config, role)
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config
        self.total_num_hidden_layers = model_config.hf_config.num_hidden_layers
        self.block_size = vllm_config.cache_config.block_size
        self.device = vllm_config.device_config.device
        self.dtype = model_config.dtype

        max_num_blocks = model_config.max_model_len * vllm_config.scheduler_config.max_num_seqs // vllm_config.cache_config.block_size

        shape = (1000, self.block_size, model_config.get_num_kv_heads(parallel_config), model_config.get_head_size()) # TODO:从config里拿到实际的blocks数量*3
        self.host_kv_cache = [
            torch.zeros(shape, dtype=self.dtype, device="cpu", pin_memory=True)
            for _ in range(self.total_num_hidden_layers)
        ]

        shape = (max_num_blocks, model_config.get_num_kv_heads(parallel_config), model_config.get_head_size())
        self.repre_cache = [
            torch.zeros(shape, dtype=self.dtype, device=self.device)
            for _ in range(self.total_num_hidden_layers)
        ]
        self.repre_pool = ReprePool(max_num_blocks)
        self.req_to_repre_blocks = dict()

        # TODO: dynamic batch_input: 在model_execute_begin的时候做当前batch的metadata准备
        # 每一次decode更新的meta: q_index, batch_offset(跟batch有关),
        # repre_index, score, score_sorted, index, index_sorted 必须每一个decode创建
        #参考model_runner, 用numpy数组，然后用H2D copy, 而不是用malloc

        # TODO: 用esa_copy来做所有的拷贝: 包括topk index


        # for esa_retrieval
        self.retrieval_input = esa_lib.RetrievalInputTensor()
        self.retrieval_input.workspace = torch.zeros(10000, dtype=torch.int32).to(self.device) # TODO: not fixed

        self.retrieval_output = esa_lib.RetrievalOutputTensor()
        self.retrieval_output.score = torch.zeros(max_num_blocks,
                                                  dtype=self.dtype,
                                                  device=self.device)
        self.retrieval_output.index = torch.zeros(max_num_blocks,
                                                  dtype=torch.int32,
                                                  device=self.device)
        self.retrieval_output.score_sorted = torch.zeros(max_num_blocks,
                                                  dtype=self.dtype,
                                                  device=self.device)
        self.retrieval_output.index_sorted = torch.zeros(max_num_blocks,
                                                  dtype=torch.int32,
                                                  device=self.device)

        self.topk_result = torch.zeros(max_num_blocks,
                                       dtype=self.dtype,
                                       device=self.device)

        # for esa_repre
        self.repre_index_cpu = torch.zeros(max_num_blocks,
                                       dtype=torch.int32,
                                       device="cpu", pin_memory=True)
        self.repre_index = torch.zeros(max_num_blocks,
                                       dtype=torch.int32,
                                       device=self.device)
        self.block_tables_cpu = torch.zeros(max_num_blocks,
                                       dtype=torch.int32,
                                       device="cpu", pin_memory=True)
        self.block_tables = torch.zeros(max_num_blocks,
                                       dtype=torch.int32,
                                       device=self.device)
        self.num_blocks_need_repre = 0

    def get_kv_cache(self, forward_context, layer_name):
        attn = forward_context.no_compile_layers[layer_name]
        kv_cache = attn.kv_cache[forward_context.virtual_engine]
        return kv_cache

    def get_layer_id(self, layer_name):
        layer_id = int(layer_name.split(".")[2])
        return layer_id

    def build_sparse_meta(
        self, scheduler_output, requests, input_batch, attn_metadata
        ):
        return
        # TODO: handle preemption
        with nvtx.range(f"build_sparse_meta"):
            if isinstance(attn_metadata, dict):
                attn_metadata = next(iter(attn_metadata.values()))
            self.attn_metadata = attn_metadata
            prefill_offset = 0
            for (req_id, num_scheduled_tokens) in scheduler_output.num_scheduled_tokens.items():
                req = requests[req_id]
                is_decode = len(req.output_token_ids) > 0 # 抢占时不成立, FIXME
                is_last_chunk = (not is_decode) and (req.num_computed_tokens + num_scheduled_tokens >= req.num_prompt_tokens)
                if is_last_chunk:
                    prompt_len = len(req.prompt_token_ids)
                    prompt_blocks = math.ceil(prompt_len / self.block_size) # 包括最后一个不满的block
                    new_blocks = self.repre_pool.allocate(prompt_blocks)
                    self.req_to_repre_blocks[req_id] = new_blocks
                    for i, b in enumerate(new_blocks):
                        self.repre_index_cpu[prefill_offset + i] = b
                    for i, b in enumerate(req.block_ids[0]):
                        self.block_tables_cpu[prefill_offset + i] = b
                    prefill_offset += prompt_blocks

            if prefill_offset > 0:
                bytes = math.ceil(prefill_offset / 8) * 8 * 4 # 对齐32bytes
                esa_copy(self.repre_index_cpu, self.repre_index, bytes)
                esa_copy(self.block_tables_cpu, self.block_tables, bytes)
            self.num_blocks_need_repre = prefill_offset

    def attention_begin(
        self,
        query,
        key,
        value,
        layer_name,
        forward_context,
        phase = None,
    ) -> None:
        return

    def attention_finished(
        self,
        query,
        key,
        value,
        attn_output,
        layer_name,
        forward_context,
        phase = None,
    ) -> None:
        return
        with nvtx.range(f"attention_finished"):
            layer_id = self.get_layer_id(layer_name)
            if self.num_blocks_need_repre > 0:
                k_cache, _ = self.get_kv_cache(forward_context, layer_name)
                esa_repre(k_cache.flatten(-2, -1), self.repre_cache[layer_id].flatten(-2, -1),
                          self.block_tables[:self.num_blocks_need_repre], self.repre_index[:self.num_blocks_need_repre])
                esa_scatter_copy(k_cache.flatten(-3), self.host_kv_cache[layer_id].flatten(-3),
                                 self.block_tables[:self.num_blocks_need_repre], self.repre_index[:self.num_blocks_need_repre])

    def estimate_num_slots_sparsed(self, request) -> int:
        return INVALID_SLOT
