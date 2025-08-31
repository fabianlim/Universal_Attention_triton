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
    # - memory O(l^2 / chunk_size)
    chunked_decay = torch.empty(
        b, h, l // chunk_size, l,
        device=q.device
    )
    for i in range(0, l // chunk_size):  
        mask = (
            torch.arange(
                i*chunk_size,(i+1)*chunk_size,
            ).unsqueeze(-1) <= 
            torch.arange(l).unsqueeze(-2) 
        ).to(q.device)
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
        ).prod(-2) # 

        # store the chunked delay
        chunked_decay[...,i,:] = decay

    # pass the chunked decay
    chunked_decay = chunked_decay.cumprod(-2)

    # can be compute in parallel using
    # l / chunks_size instances

    targ = torch.empty(b,h,l,d)
    for i in range(0, l // chunk_size):  
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


        # with strict < to the the upper triangular
        mask = (
            torch.arange(
                i*chunk_size,(i+1)*chunk_size,
            ).unsqueeze(-1) <= 
            torch.arange(l).unsqueeze(-2) 
        ).to(q.device)
        mask2 = (
            torch.arange(
                i*chunk_size,(i+1)*chunk_size,
            ).unsqueeze(-1) <
            torch.arange(l).unsqueeze(-2) 
        ).to(q.device)
        decay = decay.masked_fill(
            mask, 1
        ).cumprod(
            -2
        )

        # prod the boundary
        if i > 0:
            decay *= chunked_decay[...,i-1:i,:]

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
        targ[
            ..., 
            i*chunk_size:(i+1)*chunk_size,
            :
        ] = score.exp().matmul(v)

    return targ

    # b, h, 
    q = q.view(b, h, lq // chunk_size, chunk_size, d)
    k = k.view(b, h, lk // chunk_size, chunk_size, d)

    # Compute decay values
    decay = 1 - (
        k.matmul(k.transpose(-1,-2)).relu().pow(2) # deltanet-style decay
        * static_src.unsqueeze(-1).sigmoid() # step decay
        * static_dest.unsqueeze(-2).sigmoid() # step decay
    ).pow(
        1/3 # aggregate the 3 types of decays by geometric mean
    )


