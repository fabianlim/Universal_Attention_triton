from universal_attention_autograd import simplest_implementation
from rewrite.baseline import parallelizeable_implementation
from rewrite.kernels import chunked_decay, softmax_with_decay_fwd

from copy import deepcopy
import torch
from itertools import product
from typing import Union, Tuple

from utils_methods import (
    set_seed, random_instance, 
)

import pytest

def test_two_pass_concept(
    b: int = 1, # batch
    l: int = 32, # sequence
    h: Union[int, Tuple[int,int]] = 2, # heads
    d: int = 16, # dim
    chunk_size: int = 16,
    rtol: float = 1e-4,
    atol: float = 1e-4,
):

    set_seed()
    q, k, v, static_src, static_dest = random_instance(
        b, l, h, d, device='cpu', requires_grad=True
    )
    q2, k2, v2 = (
        deepcopy(q),
        deepcopy(k),
        deepcopy(v)
    )

    static_src2, static_dest2 = (
        deepcopy(static_src),
        deepcopy(static_dest),
    )

    # run reference
    out_ref, _ = simplest_implementation(q, k, v, static_src, static_dest)
    out = parallelizeable_implementation(
        q2, k2, v2, static_src2, static_dest2, 
        chunk_size=chunk_size
    )

    out_ref.norm().backward()
    out.norm().backward()
    
    torch.testing.assert_close(
        out_ref, out, atol=atol, rtol=rtol
    )
    for a, b in [
        (q, q2),
        (k, k2),
        (v, v2),
    ]:
        torch.testing.assert_close(
            a.grad, b.grad, atol=atol, rtol=rtol
        )

def _two_pass_routine(
    q: torch.tensor,
    k: torch.tensor,
    v: torch.tensor,
    src: torch.tensor,
    dest: torch.tensor,
    chunk_size: int,
    return_decay: bool,
):
    # Pass1: get the chunked kernel
    decay_chunks = chunked_decay(
        k, src, dest, 
        chunk_size=chunk_size
    )

    # - we then need to pass the chunks forward
    # NOTE: maybe write a kernel
    decay_chunks = decay_chunks.cumsum(-2)

    # Pass2: run the softmax
    return softmax_with_decay_fwd(
        q, k, v, 
        src, dest, 
        decay_chunks,
        return_decay=return_decay,
        chunk_size=chunk_size
    )

@pytest.mark.parametrize(
    "b,l,h,d,chunk_size", product(
        [1, 2, 4, 8, 32], # batch
        [32, 64, 100, 128, 256, 500, 1024], # seqlen
        [
            1, 4, 8, 16, 32, # mha
            (32, 8), # gqa
        ], 
        [16, 32, 64, 128], # head_dim
        [
            32, 64, 128 # chunk_size
        ]
    )
)
def test_two_pass_kernel_fwd(
    b: int, # batch
    l: int, # sequence
    h: int, # heads
    d: int, # dim
    chunk_size: int,
    atol: float = 5e-3,
    rtol: float = 1e-3,
):

    set_seed()
    q, k, v, static_src, static_dest = random_instance(
        b, l, h, d, device='cuda', requires_grad=False
    )

    q2, k2, v2 = (
        deepcopy(q),
        deepcopy(k),
        deepcopy(v)
    )

    static_src2, static_dest2 = (
        deepcopy(static_src),
        deepcopy(static_dest),
    )

    # run reference
    out_ref, _, decay_ref = simplest_implementation(
        q, k, v, static_src, static_dest,
        return_decay=True,
    )
    # - because this flips rows,cols
    decay_ref = torch.tril(decay_ref.transpose(-2,-1))

    out, decay = _two_pass_routine(
        q2, k2, v2, 
        static_src2, static_dest2, 
        return_decay=True,
        chunk_size=chunk_size
    )

    # - the upper triangular entries are not 
    # - gauranteed to be pop correctly
    decay = torch.tril(torch.exp(decay))

    g = decay_ref.shape[1] // decay.shape[1]
    if g > 1:
        # if there is gqa, the decay will be repeated
        decay_ref = decay_ref[:,::g]

    torch.testing.assert_close(
        decay_ref, decay, atol=atol, rtol=rtol,
    )

    torch.testing.assert_close(
        out_ref, out, atol=atol, rtol=rtol
    )
