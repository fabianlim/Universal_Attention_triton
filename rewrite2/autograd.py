from torch.autograd import Function
from Universal_Attention_triton.rewrite2.kernels import softmax_with_decay_fwd
from Universal_Attention_triton.rewrite2.kernels import compute_dYdQ, compute_dVdK

class UniversalAttention(Function):

    @staticmethod
    def forward(
        ctx, k, v, q, decay,
    ):
        # assume that k is already normalized

        out, attn = softmax_with_decay_fwd(q, k, v, decay)

        ctx.save_for_backward(
            k, v, q, attn,
        )
        return out

    @staticmethod
    def backward(ctx, dout):

        (
            k, v, q, attn
        ) = ctx.saved_tensors

        b, kvheads, l, d = k.shape


        dY, dQ = compute_dYdQ(
            dout, q, k, v, attn, 
        )

        dV, dK = compute_dVdK(
            dout, q, dY, attn, 
            kvheads=kvheads,
        )

        dY = dY.view(b, kvheads, -1, l, l).sum(2)
        dK = dK.view(b, kvheads, -1, l, d).sum(2)
        dV = dV.view(b, kvheads, -1, l, d).sum(2)


        # NOTE: missing one component of dK
        # - dY is the gradient for decay
        return dK, dV, dQ, dY
