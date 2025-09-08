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
        # NOTE: we should need to recomp, because now we have to 
        # go columnwise
        # - the recomp should be sufficient with just the rowwise max and denom

        # DEFINITION: let Z = softmax(...), the matrix of r x c softmax values
        # FIXME: REPLACE dV with a kernel 
        # - the kernel should just compute the scores columnwise and accumulate 
        #   the dV values
        # dV_0       * * * *  | dO_0
        # dV_1   =     * * *  | dO_1
        # dV_2           * *  | dO_2
        # dV_3             *  | dO_3
        # backprop ==>: O = Z * V => dV = dO @ Z^T
        l = decay.shape[2]
        decay = decay.masked_fill(
            torch.ones(l,l, device=k.device).triu(1).bool(),
            - torch.inf
        )
        logits = q.matmul(k.transpose(-1,-2)).add(decay)
        denom = logits.logsumexp(dim=-1)
        score = logits.sub(denom.unsqueeze(-1))
        score = score.exp()
        dV = score.transpose(-1, -2).matmul(dout) # score columnwise sum

        # FIXME: replace dQ with a kernel
        # - first we backprop through the softmax values Z = softmax(...), see above
        # backprop ==>: 0 = Z @ V => dZ = tril(dO @ V^T)
        # - where
        #                   *       
        # dZ = dO @ V^T =   * *     
        #                   * * *   
        #                   * * * * 
        # - the Jacobian of the softmax, dSoftmax, is:
        #              SM(a) * (1-SM(a))   -SM(a) SM(b)         -SM(a) SM(c)
        # dSoftmax  = -SM(b) SM(a)         SM(b) * (1-SM(b))    -SM(b) SM(c)
        #             -SM(c) * SM(a)      -SM(c) SM(b)           SM(c) * (1-SM(c))
        #            
        # backprop ==>: Z = SM(Y) => dY = dZ dSoftmax
        # - which dZ dSoftmax can be derived to be equivalent to:
        #   [dZ - (dZ * score).sum(axis=-1)] * score, here score == softmax
        # 
        # finally backprop to the Q values:
        # backprop ==>: Y = Q @ K^T + log(D).cumsum(-2) => dQ = dY @ K

        # - backprop to softmax values Z:
        dZ = dout.matmul(v.transpose(-2,-1)) # ignored the tril as it will be handled
                                             # when we mult with score below

        # - backprop through softmax to the inputs Y:
        # - simialrly, dY will be tril like dZ
        dY = dZ - (dZ * score).sum(-1, keepdim=True) # see notes above
        dY *= score

        # - backprop to Q values
        dQ = dY.matmul(k) 

        # FIXME: replace dK with a kernel
        # - sharing some initial steps with Q, we begin at 
        #   Y = Q @ K^T + log(D).cumsum(-2)

        # backprop ==>: Y = Q @ K^T + log(D).cumsum(-2) ==> dK = dY @ Q^T + dlog(D)dY
        # 
        # where here log D = log (1 + (ReLU(K * K^T).pow(2) * dest @ src).pow(1/3) )
        # are a tril of decay values, and the cumsum is applied on the rows.

        # DEFINITION: we define the following steps, 
        # - Let log(D) = Z1 and Y2 = log(D).cumsum(-2)
        # - Y2 = cumsum(Z1)
        # - Z1 = log(1-Z2.pow(1/3))             [pointwise]
        # - Z2 = dest @ src @ Z3.relu().pow(2)  [pointwise]
        # - Z3 = K @ K^T
        # and we backprop through each of them

        # backprop ==>:  Y2 = cumsum(Z1) => dZ1 = U * dY where U is a triu
        # - Since Z1 = log(D), and D.exp() has 1's on the diag
        #   and zeros on the upper triangular, thus we must have
        #          0
        #   dZ1 =  * 0
        #          * * 0
        #          * * * 0
        dZ1 = torch.ones(l,l, device=dY.device).triu().matmul(dY) # b,h,l,d
        dZ1 = dZ1.tril(-1) # take the lower tri

        # NOTE: we recomp Z2 here, but it should be saved
        Z3 = k.matmul(k.transpose(-1,-2))
        ds = dest.unsqueeze(-1).sigmoid() * src.unsqueeze(-2).sigmoid()
        Z2 = Z3.relu().pow(2) * ds

        # backprop ==>:  Z1 = log(1-Z2.pow(1/3)) => dZ2 = dZ1 * -1 / 3 / (z.pow(2/3) -z)
        # - similarly, dZ2 will be tril(-1) like dZ1
        term = 3 * (Z2.pow(2/3) - Z2)
        dZ2 = dZ1 * (- 1. / (term + 1e-6))

        # backprop ==>: Z2 = dest @ src * Z3.relu().pow(2) 
        # => dZ3 = dZ2 * dest * src * 2 * Z3, if Z3 >= 0 and 0 otherwise
        # - simialrly, dZ3 will be tril(-1) like dZ2 and dZ1
        dZ3 = torch.where(
            Z3 >= 0,
            dZ2 * ds * 2 * Z3, 0.
        ) # b,h,l,l

        # backprop ==>: Z3 = K @ K^T => dK = 2 * dZ3 @ K
        dK = 2 * dZ3.matmul(k)

        # backprop ==>: Y = Q @ K^T + log(D).cumsum(-2) ==> dK = dY @ Q^T + dlog(D)dY
        dK += dY.transpose(-2, -1).matmul(q)

        # FIXME: replace dsrc and ddest with kernel
        # - for these two we begin from Z2 = dest @ src @ Z3.relu().pow(2) 
        #   where dest is rowwise and src is columnwise

        # backprop ==>: dsrc = (dest @ Z3.relu().pow(2)).sum(-2)
        dsrc = (
            dZ2 * dest.unsqueeze(-1).sigmoid() * Z3.relu().pow(2)
        ).sum(-2)

        # backprop ==>: ddest = (src @ Z3.relu().pow(2)).sum(-1)
        ddest = (
            dZ2 * src.unsqueeze(-2).sigmoid() * Z3.relu().pow(2)
        ).sum(-1)

        # NOTE: since we accepted src and dest befor ethe sigmoid
        # we need to accomodate for the transformation, since the above
        # derivation assumed src and dest were sigmoided
        dsrc *= (
            src.sigmoid() * (1 - src.sigmoid())
        )
        ddest *= (
            dest.sigmoid() * (1 - dest.sigmoid())
        )

        return (
            dK, # k
            dV, 
            dQ, 
            dsrc, 
            ddest
        )

        