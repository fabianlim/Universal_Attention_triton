from torch.autograd import Function
from .kernels import chunked_decay, softmax_with_decay_fwd
import torch


class UniversalAttention(Function):

    @staticmethod
    def forward(
        ctx, k, v, q, src, dest,
    ):
        ## NOTE: assume k is normalized and
        # src and dest are sigmoided

        # Pass1: get the chunked kernel
        decay_chunks = chunked_decay(
            k, src, dest, 
        )

        # - we then need to pass the chunks forward
        # NOTE: maybe write a kernel
        decay_chunks = decay_chunks.cumsum(-2)

        # Pass2: run the softmax
        o, decay = softmax_with_decay_fwd(
            q, k, v, 
            src, dest, 
            decay_chunks,
            return_decay=True, # FIXME
        )

        ctx.save_for_backward(
            k, v, q, src, dest, # decay_chunks,
            decay, #FIXME
        )

        return o

    @staticmethod
    def backward(ctx, dout):
        # Note: when using mixed precision, dout is downcast but ddenom is always fp32

        (
            k, v, q, src, dest, 
            decay,
            # decay_chunks 
        ) = ctx.saved_tensors

        # decay_chunks = chunked_decay(
        #     k, dest, src, 
        # )
        # decay_chunks = decay_chunks.cumsum(-2)
        
        # # NOTE: for now we need to recomp.. but we will fuse later
        # dv = softmax_with_decay_fwd(
        #     k, q, dout, 
        #     dest, src, 
        #     decay_chunks,
        # )

        # Defn: let Z = softmax(...)

        # FIXME: dV
        # O = Z * V => dV = dO * Z^T
        l = decay.shape[2]
        decay = decay.masked_fill(
            torch.ones(l,l, device=k.device).triu(1).bool(),
            - torch.inf
        )
        logits = q.matmul(k.transpose(-1,-2)).add(decay)
        denom = logits.logsumexp(dim=-1)
        score = logits.sub(denom.unsqueeze(-1))
        score = score.exp()
        dV = score.transpose(-1, -2).matmul(dout)

        # FIXME: dQ
        # 0 = Z * V => dZ = dO * V^T 
        # Z = SM(Y) => dY = dZ * dSoftmax
        # Y = QK^T + log(D) => dQ = dY * K

        dZ = dout.matmul(v.transpose(-2,-1))

        # - the next two lines are equivalent to right
        #   multiplying by dSoftmax
        #   (takes some derivation)
        dY = dZ - (dZ * score).sum(-1, keepdim=True)
        dY *= score

        # - get the output
        dQ = dY.matmul(k) 

        return (
            None, # k
            dV, 
            dQ, 
            None, None
        )

        