import torch
from torch.autograd import Function

# thanks @shawntan for pointing to me ideas in the 
# https://github.com/zhixuan-lin/forgetting-transformer/blob/main/src/forgetting_transformer/ops/forgetting_attention.py#L628
def parallelizeable_implementation(
    q: torch.Tensor, # b,h,l,d
    k: torch.Tensor, # b,h,l,d
    v: torch.Tensor, # b,h,l,d
    static_src: torch.Tensor,
    static_dest: torch.Tensor,
    chunk_size: int = 16,
):

    # b, h, l, d
    b, h, l, d = q.shape
    _, hkv, _, _ = k.shape
    assert l % chunk_size == 0, \
        f"this function works for queries in multiple of {chunk_size}"

    # L2-normalize K
    k = k/k.pow(2).sum(-1,True).sqrt().add(1e-6)

    def compute_decay(
        kr, # k-row
        kc, # k-col
        src, # source
        dest, # dest
    ):
        # Compute decay values
        decay = 1 - (
            kr.matmul(kc.transpose(-1,-2)).relu().pow(2) # deltanet-style decay
            * dest.unsqueeze(-1).sigmoid() # step decay
            * src.unsqueeze(-2).sigmoid() # step decay
        ).pow(
            1/3 # aggregate the 3 types of decays by geometric mean
        )
        return decay

    # can be compute in parallel using
    # l / chunks_size instances
    chunked_decay = torch.empty(
        b, hkv, l // chunk_size, l,
        device=q.device
    )
    for i in range(0, l // chunk_size):  

        # - compute the mask for the chunked rows
        mask = (
            torch.arange(
                i*chunk_size,(i+1)*chunk_size,
            ).unsqueeze(-1) <= 
            torch.arange(l).unsqueeze(-2) 
        ).to(q.device)

        # - compute the delay, product across the chunk
        # - we only need the final prod, do not
        #   need to store all O(l/chunk_size) forws
        #   of delay values
        decay = compute_decay(
            k[
                ..., 
                i*chunk_size:(i+1)*chunk_size,
                :
            ],
            k,
            static_src,
            static_dest[
                ...,
                i*chunk_size:(i+1)*chunk_size,
            ],
        ).masked_fill(
            mask, 1
        ).prod(-2) # we only need the final prod

        # store the chunked delay
        chunked_decay[...,i,:] = decay

    # pass the chunked decay
    # - similar to mamba state passing
    chunked_decay = chunked_decay.cumprod(-2)

    # expand heads
    if hkv < h:
        k = torch.repeat_interleave(k, h // hkv, dim=1)
        v = torch.repeat_interleave(v, h // hkv, dim=1)
        chunked_decay = torch.repeat_interleave(chunked_decay, h // hkv, dim=1)
        static_src = torch.repeat_interleave(static_src, h // hkv, dim=1)
        static_dest = torch.repeat_interleave(static_dest, h // hkv, dim=1)

    # can be computed in parallel using
    # l / chunks_size instances
    outputs = []
    for i in range(0, l // chunk_size):  

        # - compute the delay across the chunked rows
        # - this can be fused as we recompute the delay
        decay = compute_decay(
            k[
                ..., 
                i*chunk_size:(i+1)*chunk_size,
                :
            ],
            k,
            static_src,
            static_dest[
                ...,
                i*chunk_size:(i+1)*chunk_size,
            ],
        )

        # - part of recomputation: chunk mask
        mask = (
            torch.arange(
                i*chunk_size,(i+1)*chunk_size,
            ).unsqueeze(-1) <= 
            torch.arange(l).unsqueeze(-2) 
        ).to(q.device)

        # recomputation: cumprod
        decay = decay.masked_fill(
            mask, 1
        ).cumprod(
            -2
        )

        # take into account the chunked boundary
        # conditions
        if i > 0:
            decay = decay * chunked_decay[...,i-1:i,:]

        # - the the causal delay values
        # with strict < to the the upper triangular
        mask2 = (
            torch.arange(
                i*chunk_size,(i+1)*chunk_size,
            ).unsqueeze(-1) <
            torch.arange(l).unsqueeze(-2) 
        ).to(q.device)
        decay = decay.masked_fill(mask2, 0)

        # get the row-chunked logits
        # - can be computed by online softmax
        logits = q[
            ..., 
            i*chunk_size:(i+1)*chunk_size,
            :
        ].matmul(
            k.transpose(-1,-2)
        ).add(decay.log())
        denom = logits.logsumexp(dim=-1)
        score = logits.sub(denom.unsqueeze(-1))

        # collect the outputs
        outputs.append(score.exp().matmul(v))

    return torch.concat(outputs, dim=-2)

def softmax(x):
    denom = x.logsumexp(dim=-1)
    score = x.sub(denom.unsqueeze(-1))
    out = score.exp()
    return out

# just a softmax for testing
class Softmax(Function):

    @staticmethod
    def forward(
        ctx, x,
    ):
        out = softmax(x)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, dout):
        # Note: when using mixed precision, dout is downcast but ddenom is always fp32

        S, = ctx.saved_tensors 
        # D = - S.outer(S)
        # D = torch.masked_fill(D, torch.eye(l, dtype=bool), 0)
        # D += ( torch.diag(S) * (1-torch.diag(S)))
        # out = dout.matmul(D)

        # equiv to
        out = (dout - (dout * S).sum()) * S
        print (out)
        return out