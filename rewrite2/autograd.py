from torch.autograd import Function
from .kernels import softmax_with_decay_fwd

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
        pass