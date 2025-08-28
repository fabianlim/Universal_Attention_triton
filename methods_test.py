import torch
from copy import deepcopy

from utils_methods import set_seed, random_instance, blockwise_ua_test_harness
from universal_attention_autograd import simplest_implementation

def test_ua_slow(
    b: int = 1, # batch
    l: int = 32, # sequence
    h: int = 2, # heads
    d: int = 4, # dim
    rtol: float = 1e-4,
    atol: float = 1e-4,
):
    set_seed()
    q, k, v, static_src, static_dest = random_instance(
        b, l, h, d, device='cuda', requires_grad=True
    )
    q2, k2, v2 = (
        deepcopy(q),
        deepcopy(k),
        deepcopy(v)
    )

    # run reference
    out_ref, denom_ref = simplest_implementation(q, k, v, static_src, static_dest)

    # run the slow kernel
    out, denom = blockwise_ua_test_harness(q2, k2, v2, static_src, static_dest, chunk_size=l)
    out = out.squeeze(-1).squeeze(-3) # b, h, l, d
    denom = denom.squeeze(-1).squeeze(-2) # b, h, l

    # backward
    out_ref.norm().backward()
    out.norm().backward()

    for a, b in [
        (out_ref, out),
        (denom_ref, denom),
        (q.grad, q2.grad),
        (k.grad, k2.grad),
        (v.grad, v2.grad),
    ]:
        torch.testing.assert_close(
            a, b, atol=atol, rtol=rtol
        )

    