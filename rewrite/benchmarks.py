
import torch
from itertools import product
from utils_methods import (
    set_seed, random_instance, 
)

from methods_test import blockwise_ua_test_harness
from Universal_Attention.triton.universal_attention_kernel import _universal_attention_fwd, _universal_attention_bwd
from kernels import (
    chunked_decay,
    softmax_with_decay_fwd
)

import json
import time

# inspired by https://github.com/IST-DASLab/marlin/blob/master/bench.py
def benchmark(f, warmup=1, iter=10):
    for i in range(warmup + iter):
        outputs = f()
        # We do not synchronize here in order to hide the kernel launch overhead during benchmarkining as this will also
        # happen during realistic model inference as many launches are submitted to the kernel queue.
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(1.)
    return res, outputs

def prepare_problem(
    b: int, l: int, h: int, d: int,
    seed: int = 42,
    **kwargs,
):
    set_seed(seed)
    q, k, v, static_src, static_dest = random_instance(
        b, l, h, d, 
        **kwargs,
    )

    # sigmoid the statics
    static_src = static_src.sigmoid() # step decay
    static_dest = static_dest.sigmoid() # step decay

    # Normalize keys
    k = k / k.pow(2).sum(-1, True).sqrt().add(1e-6)

    torch.cuda.synchronize()
    return q, k, v, static_src, static_dest

def run_legacy_impl_one(
    queries, 
    keys, 
    values, 
    static_src, static_dest,
    chunk_size,
):
    batch_size, nheads, q_len, emb_kq_per_head = queries.shape
    batch_size, kvheads, _, _ = keys.shape
    _, _, _, emb_v_per_head = values.shape

    # needs to 
    queries = queries.view(
        batch_size, 
        kvheads, 
        -1, 
        q_len, 
        emb_kq_per_head
    )  # b h r l d

    num_chunks = q_len // chunk_size
    s = [batch_size, kvheads, num_chunks, chunk_size, -1]
    kc = keys.view(*s)  # b h n c d
    vc = values.view(*s)
    static_src = static_src.view(batch_size, kvheads, num_chunks, chunk_size)  # b h n c
    # Shrink _c to 64
    static_dest = static_dest.view(batch_size, kvheads, num_chunks*2, chunk_size//2)  # b h n c
    queries = queries.view(
        batch_size, kvheads, 
        nheads // kvheads, 
        num_chunks*2, chunk_size//2, -1
    )
    out, denom = _universal_attention_fwd(kc, vc, queries, static_src, static_dest)
    return out

def run_two_pass(
    queries, 
    keys, 
    values, 
    static_src, static_dest,
    chunk_size,
):
    # Pass1: get the chunked kernel
    decay_chunks = chunked_decay(keys, static_src, static_dest)

    # - still no kernel for this
    decay_chunks = decay_chunks.cumsum(-2)

    out = softmax_with_decay_fwd(
        queries, 
        keys, 
        values, 
        static_src, static_dest,
        decay_chunks,
        return_decay=False,
        chunk_size=chunk_size,
    )
    return out

if __name__ == '__main__':

    BATCHES = [1, 2]
    SEQUENCE_LENS = [32, 64]
    HEADS = [1, 2]
    HEAD_DIM = [16, 32]
    CHUNK_SIZE = [32]

    for b, l, h, d, chunk_size in product(
        BATCHES, SEQUENCE_LENS, 
        HEADS, HEAD_DIM,
        CHUNK_SIZE
    ):
        # for now we just test the forwards
        q, k, v, static_src, static_dest = prepare_problem(
            b, l, h, d, device='cuda', 
            requires_grad=False
        )

        with torch.no_grad():
            t, _ = benchmark(lambda: run_legacy_impl_one(q, k, v, static_src, static_dest, chunk_size))
            res_l1 = {
                'b': b, 'l': l, 'h': h, 'd': d, 'chunk_size': chunk_size, 'time': t,
                'method': 'legacy_impl_one',
            }
            print (json.dumps(res_l1))

            t, _ = benchmark(lambda: run_two_pass(q, k, v, static_src, static_dest, chunk_size))
            res_tp = {
                'b': b, 'l': l, 'h': h, 'd': d, 'chunk_size': chunk_size, 'time': t,
                'method': 'two_pass',
            }
            print (json.dumps(res_tp))






