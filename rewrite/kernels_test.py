from universal_attention_autograd import simplest_implementation
from rewrite.baseline import parallelizeable_implementation
from rewrite.kernels import chunked_decay, softmax_with_decay_fwd
from rewrite.autograd import UniversalAttention as UA
from universal_attention_autograd import UniversalAttention as UA_ref

from copy import deepcopy
import torch
from itertools import product
from typing import Union, Tuple

from unittest.mock import patch

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

def _two_pass_routine_fwd(
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
def test_two_pass_kernel_fwd_and_decay(
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

    out, decay = _two_pass_routine_fwd(
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
def test_two_pass_autograd_with_simp_impl(
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
        b, l, h, d, device='cuda', 
        requires_grad=True
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

    # for the static's we need to 
    # set the grad like this because there are some
    # operations applied when creating these tensors
    static_src = static_src.requires_grad_()
    static_dest = static_dest.requires_grad_()
    static_src2 = static_src2.requires_grad_()
    static_dest2 = static_dest2.requires_grad_()

    # run reference
    out_ref, _ = simplest_implementation(
        q, k, v, static_src, static_dest,
    )
    out_ref.norm().backward()

    with patch('rewrite.kernels.CHUNK_SIZE', chunk_size):
        # the autograd function assumes k, src, dest
        # already normalized
        out = UA.apply(
            k2 / k2.pow(2).sum(-1, True).sqrt().add(1e-6),
            v2, q2,
            static_src2.sigmoid(), 
            static_dest2.sigmoid(),
        )
        out.norm().backward()

    torch.testing.assert_close(
        out_ref, out, atol=atol, rtol=rtol
    )

    # NOTE: actually the static_src and static_dest
    # grads will be nan's here for some reason
    for a, b in [
        (q.grad, q2.grad),
        (v.grad, v2.grad),
        (k.grad, k2.grad),
        (static_src.grad, static_src2.grad),
        (static_dest.grad, static_dest2.grad),
    ]:
        torch.testing.assert_close(
            a.grad, b.grad, 
            atol=atol, rtol=rtol,
        )


# harness for converting a random instance and testing it on a particular method
def _col_chunk_reference_autograd(
    queries: torch.Tensor, # b,h,l,d
    keys: torch.Tensor, # b,h,l,d
    values: torch.Tensor, # b,h,l,d
    static_src: torch.Tensor, # b,h,l
    static_dest: torch.Tensor, # b,h,l
    chunk_size: int = 32,
):
    batch_size, nheads, q_len, hdim = queries.shape
    batch_size, kvheads, _, _ = keys.shape
    # _, _, _, _ = values.shape

    # this autograd function expects chunking of the columns
    
    # Chunk inputs
    num_chunks = q_len // chunk_size
    s = [batch_size, kvheads, num_chunks, chunk_size, -1]
    kc = keys.view(*s)  # b h n c d
    vc = values.view(*s)

    # # Blockwise universal attention
    queries = queries.view(
        batch_size, 
        kvheads, 
        -1, 
        q_len,
        hdim
    )  # b h r l d
    
    static_src = static_src.view(batch_size, kvheads, num_chunks, chunk_size)  # b h n c
   
    output, denom, _ = UA_ref.apply(
        kc, vc, queries, static_src, static_dest
    )

    output = output.mul(
        denom.softmax(dim=-1).unsqueeze(-2)
    ).sum(-1) 
    return output.view(
        batch_size, 
        nheads,
        q_len, hdim
    )


@pytest.mark.parametrize(
    "b,l,h,d,chunk_size", product(
        [1, 2, 4, 8, 32], # batch
        [32, 64, 128, 256, 1024], # seqlen
        [
            1, 4, 8, 16, 32, # mha
        ], 
        [16, 32, 64, 128], # head_dim
        [
            32, # chunk_size
        ]
    )
)
def test_two_pass_autograd_with_kernel_impl(
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
        b, l, h, d, device='cuda', 
        requires_grad=True
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

    # for the static's we need to 
    # set the grad like this because there are some
    # operations applied when creating these tensors
    static_src = static_src.requires_grad_()
    static_dest = static_dest.requires_grad_()
    static_src2 = static_src2.requires_grad_()
    static_dest2 = static_dest2.requires_grad_()

    out_ref = _col_chunk_reference_autograd(
        q, 
        k / k.pow(2).sum(-1, True).sqrt().add(1e-6),
        v, 
        static_src.sigmoid(), static_dest.sigmoid(),
        chunk_size=chunk_size
    )
    out_ref.norm().backward()

    with patch('rewrite.kernels.CHUNK_SIZE', chunk_size):
        # the autograd function assumes k, src, dest
        # already normalized
        out = UA.apply(
            k2 / k2.pow(2).sum(-1, True).sqrt().add(1e-6),
            v2, q2,
            static_src2.sigmoid(), 
            static_dest2.sigmoid(),
        )
        out.norm().backward()

    torch.testing.assert_close(
        out_ref, out, atol=atol, rtol=rtol
    )

    for a, b in [
        (q.grad, q2.grad),
        (v.grad, v2.grad),
        (k.grad, k2.grad),
        (static_src.grad, static_src2.grad),
        (static_dest.grad, static_dest2.grad),
    ]:
        torch.testing.assert_close(
            a.grad, b.grad, 
            atol=atol, rtol=rtol,
        )
