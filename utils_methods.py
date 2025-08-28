import torch
import math
import random

from Universal_Attention.universal_attention import UniversalAttention
from Universal_Attention.triton.universal_attention_kernel_opt import _attention
# import numpy as np

# set random seed
def set_seed(seed: int=42):
    torch.manual_seed(seed)
    # np.random.seed(seed_value)
    random.seed(seed)

# create random instance
def random_instance(
    b: int, l: int, h: int, d: int,
    max_ = math.log(.1),
    min_ = math.log(.001),
    device = torch.device('cuda'),
    requires_grad: bool = False,
):
    q = torch.randn(b,h,l,d, device=device, requires_grad=requires_grad)
    k = torch.randn(b,h,l,d, device=device, requires_grad=requires_grad)
    v = torch.randn(b,h,l,d, device=device, requires_grad=requires_grad)

    b1 = torch.rand(h, device=device) * (max_-min_) + min_
    b2 = torch.rand(h, device=device) * (max_-min_) + min_
    static_src = torch.randn(b,h,l, device=device).add(b1.unsqueeze(-1))
    static_dest = torch.randn(b,h,l, device=device).add(b2.unsqueeze(-1))

    return q, k, v, static_src, static_dest

# harness for converting a random instance and testing it on a particular method
# - it does padding to the chunk size
def blockwise_ua_test_harness(
    queries: torch.Tensor, # b,h,l,d
    keys: torch.Tensor, # b,h,l,d
    values: torch.Tensor, # b,h,l,d
    static_src: torch.Tensor, # b,h,l
    static_dest: torch.Tensor, # b,h,l
    chunk_size: int = 128,
):
    batch_size, nheads, q_len, emb_kq_per_head = queries.shape
    batch_size, kvheads, _, _ = keys.shape
    _, _, _, emb_v_per_head = values.shape

    # sigmoid the statics
    static_src = static_src.sigmoid() # step decay
    static_dest = static_dest.sigmoid() # step decay

    # Blockwise universal attention
    queries = queries.view(
        batch_size, 
        kvheads, 
        -1, 
        q_len, 
        emb_kq_per_head
    )  # b h r l d

    # Normalize keys
    keys = keys / keys.pow(2).sum(-1, True).sqrt().add(1e-6)

    # Right-pad k,v,src if len not divisible by chunksize
    if q_len % chunk_size != 0:
        slack = chunk_size - q_len % chunk_size
        queries = torch.cat([
            queries, 
            torch.zeros(
                batch_size, 
                kvheads, 
                nheads // kvheads, 
                slack, 
                emb_kq_per_head,
                device=queries.device, 
                dtype=queries.dtype
            )], 
            dim=-2
        )
        keys = torch.cat([
                keys, 
                torch.zeros(
                    batch_size, kvheads, slack, 
                    emb_kq_per_head, 
                    device=keys.device, 
                    dtype=keys.dtype
            )], 
            dim=-2
        )
        values = torch.cat([
                values, 
                torch.zeros(
                    batch_size, kvheads, slack, 
                    emb_v_per_head, 
                    device=values.device, 
                    dtype=values.dtype
            )], 
            dim=-2
        )
        static_src = torch.cat([
                static_src, 
                torch.zeros(
                    batch_size, kvheads, slack,
                    device=static_src.device, 
                    dtype=static_src.dtype
            )], 
            dim=-1
        )
        static_dest = torch.cat([
            static_dest, 
            torch.zeros(
                batch_size, kvheads, slack,
                device=static_dest.device, 
                dtype=static_dest.dtype
            )], 
            dim=-1
        )

    # Chunk inputs
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
            
    # Inputs:
    # kc: b h n_ c_ d
    # vc: b h n_ c_ d
    # xq: b h r _n _c d
    # static_src: b h n_ c_
    # static_dest: b h _n _c
    output, denom = UniversalAttention.apply(
        kc, vc, queries, static_src, static_dest
    )

    return output, denom

# - it runs the optimized 
def blockwise_ua_test_harness2(
    queries: torch.Tensor, # b,h,l,d
    keys: torch.Tensor, # b,h,l,d
    values: torch.Tensor, # b,h,l,d
    static_src: torch.Tensor, # b,h,l
    static_dest: torch.Tensor, # b,h,l
):
    # batch_size, nheads, q_len, emb_kq_per_head = queries.shape
    # batch_size, kvheads, _, _ = keys.shape
    # _, _, _, emb_v_per_head = values.shape

    # need to do this because the kernel has some wierd hacks
    static_src.requires_grad_()
    static_dest.requires_grad_()

    # sigmoid the statics
    static_src = static_src.sigmoid() # step decay
    static_dest = static_dest.sigmoid() # step decay

    # Normalize keys
    keys = keys / keys.pow(2).sum(-1, True).sqrt().add(1e-6)

    output = _attention.apply(
        queries,
        keys,
        values,
        True, 1.0, 
        static_src,
        static_dest,
    )
    return output, None