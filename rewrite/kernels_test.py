from universal_attention_autograd import simplest_implementation
from rewrite.baseline import parallelizeable_implementation
from rewrite.kernels import chunked_decay, softmax_with_decay_fwd

from copy import deepcopy
import torch
from itertools import product

from utils_methods import (
    set_seed, random_instance, 
)

import pytest

def test_two_pass_implementation(
    b: int = 1, # batch
    l: int = 32, # sequence
    h: int = 2, # heads
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


@pytest.mark.parametrize(
    "b,l,h,d,chunk_size", product(
        [1, 2, 4, 8, 32],
        [32, 64, 128, 256, 1024],
        [1, 4, 8, 16, 32],
        [16, 32, 64, 128],
        [
            16, # 32, 64, 128
        ]
    )
)
def test_two_pass_kernel_fwd(
    b: int, # batch
    l: int, # sequence
    h: int, # heads
    d: int, # dim
    chunk_size: int,
    atol: float = 1e-2,
    rtol: float = 1e-3,
):

    if l % chunk_size != 0:
        pytest.skip(
            f"sequence length {l} is not divisible by {chunk_size}"
        )

    set_seed()
    q, k, v, static_src, static_dest = random_instance(
        b, l, h, d, device='cuda', requires_grad=True
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

    # Pass1: get the chunked kernel
    decay_chunks = chunked_decay(k, static_src, static_dest)

    # - we then need to pass the chunks forward
    # NOTE: maybe write a kernel
    decay_chunks = decay_chunks.cumsum(-2)

    # Pass2: run the softmax
    out, decay = softmax_with_decay_fwd(
        q2, k2, v2, 
        static_src2, static_dest2, 
        decay_chunks,
        return_decay=True,
        chunk_size=chunk_size
    )
    # - the upper triangular entries are not 
    # - gauranteed to be pop correctly
    decay = torch.tril(torch.exp(decay))

    torch.testing.assert_close(
        decay_ref, decay, atol=atol, rtol=rtol,
    )

    torch.testing.assert_close(
        out_ref, out, atol=atol, rtol=rtol
    )
