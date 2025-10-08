
import torch
from itertools import product
from utils_methods import (
    set_seed, random_instance, 
)

from Universal_Attention.universal_attention import UniversalAttention as UA_ref1
from Universal_Attention.triton.universal_attention_kernel_opt import _attention
from autograd import UniversalAttention as UA

from unittest.mock import patch
from contextlib import nullcontext

from functools import partial
import json
import time
from typing import Union, Callable, Tuple

# inspired by https://github.com/IST-DASLab/marlin/blob/master/bench.py
def benchmark(
    f, 
    args: Union[Tuple, Callable],
    warmup=1, iter=10, 
    run_backward: bool = False,
    account_for_args: bool = False,
):

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_reserved = torch.cuda.memory_reserved()
    mem_allocated = torch.cuda.memory_allocated()

    args_time = 0.0 # account for args creation time
    tick = time.time()
    for i in range(warmup + iter):

        if run_backward:
            # we need to call to create the args
            # - this will eat into some of the estimated time
            args_tick = time.time()
            _args = args()
            if account_for_args and i >= warmup:
                print ('args time')
                args_time += (time.time() - args_tick)
        else:
            _args = args

        try:
            outputs = f(*_args)
            if run_backward:
                outputs.norm().backward()
        except Exception as e: 
            return {
                'exception': str(e)
            }, None
        # We do not synchronize here in order to hide the kernel launch overhead during benchmarkining as this will also
        # happen during realistic model inference as many launches are submitted to the kernel queue.
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = max(time.time() - tick - args_time, 0) / iter
    mem_reserved = torch.cuda.max_memory_reserved() - mem_reserved 
    mem_allocated = torch.cuda.max_memory_allocated() - mem_allocated
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(1.)
    return {
        'time': res,
        'mem_reserved': mem_reserved,
        'mem_allocated': mem_allocated,
    }, outputs

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
    output, denom = UA_ref1.apply(
        kc, vc, queries, static_src, static_dest
    )
    output = output.mul(
        denom.softmax(dim=-1).unsqueeze(-2)
    ).sum(-1) 
    return output

def run_legacy_impl_two(
    queries, 
    keys, 
    values, 
    static_src, static_dest,
):
    output = _attention.apply(
        queries,
        keys,
        values,
        True, 1.0, 
        static_src,
        static_dest,
    )
    return output

def run_two_pass(
    queries, 
    keys, 
    values, 
    static_src, static_dest,
    chunk_size,
):
    with patch('rewrite.kernels.CHUNK_SIZE', chunk_size):
        out = UA.apply(
            keys / keys.pow(2).sum(-1, True).sqrt().add(1e-6),
            values, queries,
            static_src.sigmoid(), 
            static_dest.sigmoid(),
        )
    return out

if __name__ == '__main__':

    BATCHES = [
        1, 8, 16,
    ]
    SEQUENCE_LENS = [
        32, 500, 1024
    ]
    HEADS = [
        4, 16, 32
    ]
    HEAD_DIM = [
        16, 32, 128
    ]
    CHUNK_SIZE = [
        128 # 32, 64, 128
    ]
    RUN_BACKWARD = [
        # False, 
        True
    ]

    for b, l, h, d, chunk_size, run_backward in product(
        BATCHES, SEQUENCE_LENS, 
        HEADS, HEAD_DIM,
        CHUNK_SIZE,
        RUN_BACKWARD
    ):

        if run_backward:
            ctx = nullcontext

            # we have to pass the function in to create the 
            # problem inside the benchmark function
            def args():
                q, k, v, src, dest = prepare_problem(
                    b, l, h, d,
                    device='cuda', requires_grad=True
                )
                src = src.requires_grad_()
                dest = dest.requires_grad_()
                return q, k, v, src, dest
        else:
            ctx = torch.no_grad
            args = prepare_problem(
                b, l, h, d, device='cuda', 
                requires_grad=run_backward
            )

        with ctx():

            if l % chunk_size == 0:
                res, _ = benchmark(
                    f=partial(run_legacy_impl_one, chunk_size=chunk_size),
                    args=args,
                    run_backward=run_backward,
                )
                res_l1 = {
                    'b': b, 'l': l, 'h': h, 'd': d, 'chunk_size': chunk_size, 
                    **res,
                    'method': 'legacy_impl_one',
                    'run_backward': run_backward,
                }
                print (json.dumps(res_l1), flush=True)

            res, _ = benchmark(
                f=run_legacy_impl_two,
                args=args,
                run_backward=run_backward,
            )
            res_l2 = {
                'b': b, 'l': l, 'h': h, 'd': d, 'chunk_size': chunk_size, 
                **res,
                'method': 'legacy_impl_two',
                'run_backward': run_backward,
            }
            print (json.dumps(res_l2), flush=True)

            res, _ = benchmark(
                f=partial(run_two_pass, chunk_size=chunk_size),
                args=args,
                run_backward=run_backward,
            )
            res_tp = {
                'b': b, 'l': l, 'h': h, 'd': d, 'chunk_size': chunk_size, 
                **res,
                'method': 'two_pass',
                'run_backward': run_backward,
            }
            print (json.dumps(res_tp), flush=True)

    print ("Benchmark completed!")






