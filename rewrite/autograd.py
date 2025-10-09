from torch.autograd import Function
from Universal_Attention_triton.rewrite.kernels import (
    chunked_decay, softmax_with_decay_fwd, 
    rowwise_bwd, colwise_bwd,
)
import torch

def _backward_slow_draft(
    dout, k, v, q, src, dest, 
    decay_chunks, # unused
    score_denom, # unused
):
    # L2-normalize K
    k = k/k.pow(2).sum(-1,True).sqrt().add(1e-6)

    # ----- START OF RECOMP (all this should go away) -----
    # - the forward pass should give the 
    # 1. chunked-delays
    # 2. denom

    # NOTE: we recomp Z2 here, but it should be saved
    Z3 = k.matmul(k.transpose(-1,-2))
    ds = dest.unsqueeze(-1) * src.unsqueeze(-2)
    Z2 = Z3.relu().pow(2) * ds
    decay = torch.log(1 - Z2.pow(1/3))

    l = decay.shape[2]
    decay = decay.masked_fill(
        torch.ones(l,l, device=k.device).triu().bool(),
        0.  # over the seq dimenstion
    ).cumsum(
        -2  # 
    ).masked_fill(
        torch.ones(l,l, device=k.device).triu(1).bool(),
        - torch.inf
    )
    
    logits = q.matmul(k.transpose(-1,-2)).add(decay)
    denom = logits.logsumexp(dim=-1)
    score = logits.sub(denom.unsqueeze(-1))
    score = score.exp()

    # --- End of RECOMP -----

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

    # DEBUG
    # daff = -dZ1.transpose(-2, -1) # line 150
    # aff2 = Z3.relu().pow(2/3)
    # # dstat = daff * aff2 
    # # dstat_src = (dstat * dest.unsqueeze(-2).pow(1/3)).sum(-1)
    # dstat = -dZ1 * Z3.relu().pow(2/3)
    # dstat_src = (dstat * dest.unsqueeze(-1).pow(1/3)).sum(-2)
    # dstat_src /= (3 * src.pow(2/3))

    # FIXME: replace dsrc and ddest with kernel
    # - for these two we begin from Z2 = dest @ src @ Z3.relu().pow(2) 
    #   where dest is rowwise and src is columnwise

    # backprop ==>: dsrc = (dest @ Z3.relu().pow(2)).sum(-2)
    dsrc = (
        dZ2 * dest.unsqueeze(-1) * Z3.relu().pow(2)
    ).sum(-2)

    # backprop ==>: ddest = (src @ Z3.relu().pow(2)).sum(-1)
    ddest = (
        dZ2 * src.unsqueeze(-2) * Z3.relu().pow(2)
    ).sum(-1)

    return (
        dK, # k
        dV, 
        dQ, 
        dsrc, 
        ddest
    )

def _backward_slow_parallelizable(
    dout, k, v, q, src, dest, chunked_decay,
    score_denom,
    chunk_size=32,
):
    # L2-normalize K
    k = k/k.pow(2).sum(-1,True).sqrt().add(1e-6)

    b, h, l, d = q.shape
    _, hkv, _, _ = k.shape
    
    def _compute_decay(
        kr, # k-row
        kc, # k-col
        src, # source
        dest, # dest
        return_deltanet = False,
    ):
        # Compute decay values
        deltanet = kr.matmul(kc.transpose(-1,-2)) 
        decay = 1 - (
            deltanet.relu().pow(2) # deltanet-style decay
            * dest.unsqueeze(-1) * src.unsqueeze(-2)
        ).pow(
            1/3 # aggregate the 3 types of decays by geometric mean
        )

        if return_deltanet:
            return torch.log(decay), deltanet
        return torch.log(decay)

    # ROWWISE!
    # can be computed in parallel using
    # l / chunks_size instances
    dZscore_store = []
    outputs_dQ = []
    for i in range(0, l // chunk_size):  
        # - compute the delay across the chunked rows
        # - this can be fused as we recompute the delay
        decay = _compute_decay(
            k[
                ..., 
                i*chunk_size:(i+1)*chunk_size,
                :
            ],
            k,
            src,
            dest[
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

        # recomputation: cumsum
        decay = decay.masked_fill(
            mask, 0.
        ).cumsum(-2)

        # take into account the chunked boundary
        # conditions
        if i > 0:
            decay = decay + chunked_decay[...,i-1:i,:]

        # - the the causal delay values
        # with strict < to the the upper triangular
        mask2 = (
            torch.arange(
                i*chunk_size,(i+1)*chunk_size,
            ).unsqueeze(-1) <
            torch.arange(l).unsqueeze(-2) 
        ).to(q.device)
        decay = decay.masked_fill(mask2, - torch.inf)

        # get the row-chunked logits
        # - can be computed by online softmax
        logits = q[
            ..., 
            i*chunk_size:(i+1)*chunk_size,
            :
        ].matmul(
            k.transpose(-1,-2)
        ).add(decay)
        denom = score_denom[
            ..., 
            i*chunk_size:(i+1)*chunk_size,
        ]
        score = logits.sub(denom.unsqueeze(-1))
        score = score.exp()

        # dZ
        dZ = dout[
            ..., 
            i*chunk_size:(i+1)*chunk_size,
            :
        ].matmul(
            v.transpose(-1,-2)
        )
        _dzScore = (dZ * score).sum(-1, keepdim=True) # see notes above
        dY = dZ - _dzScore
        dY *= score

        dZscore_store.append(_dzScore) # store it for later

        # - backprop to Q values
        outputs_dQ.append(dY.matmul(k))


    # from the column summaries
    dZscore_store = torch.concat(dZscore_store, dim=-2)
    outputs_dV = []
    outputs_dK_1 = []
    outputs_dK_2 = []
    outputs_dsrc = []
    outputs_ddest = []

    # COLWISE!
    for j in range(0, l // chunk_size):  
        # - compute the delay for a chunked column
        decay, deltanet = _compute_decay(
            k,
            k[
                ..., 
                j*chunk_size:(j+1)*chunk_size,
                :
            ],
            src[
                ...,
                j*chunk_size:(j+1)*chunk_size,
            ],
            dest,
            return_deltanet=True
        )
        deltanet_relu2 = deltanet.relu().pow(2)

        # - part of recomputation: chunk mask
        mask = (
            torch.arange(l).unsqueeze(-1) <=
            torch.arange(
                j*chunk_size,(j+1)*chunk_size,
            ).unsqueeze(-2) 
        ).to(q.device)

        # recomputation: cumsum
        decay = decay.masked_fill(
            mask, 0.
        ).cumsum(-2)

        mask2 = (
            torch.arange(l).unsqueeze(-1) <
            torch.arange(
                j*chunk_size,(j+1)*chunk_size,
            ).unsqueeze(-2) 
        ).to(q.device)
        decay = decay.masked_fill(mask2, - torch.inf)

        # get the column chunk logits
        logits = q.matmul(
            k[
                ..., 
                j*chunk_size:(j+1)*chunk_size,
                :
            ].transpose(-1,-2)
        ).add(decay)

        # take the denom from store
        score = logits.sub(
            score_denom.unsqueeze(-1)
        )
        score = score.exp()
        
        # - dV
        outputs_dV.append(
            score.transpose(-1, -2).matmul(dout)
        )

        # get the column chunk logits
        dZ = dout.matmul(
            v[
                ..., 
                j*chunk_size:(j+1)*chunk_size,
                :
            ].transpose(-1,-2)
        )
        dY = dZ - dZscore_store # take from store
        dY *= score

        # compute dZ1 which is row-sum of Y
        dZ1 = torch.where(
            mask == False, # negate it so its i > j
            dY.flip(-2).cumsum(dim=-2).flip(-2),
            0.
        )

        # Z2 = Z3.relu().pow(2) * ds
        ds = (
            src[
                ..., 
                j*chunk_size:(j+1)*chunk_size,
            ].unsqueeze(-2)
            * dest.unsqueeze(-1)
        )
        term = deltanet_relu2 * ds # Z2
        term = term.pow(2/3) - term
        dZ2 = - dZ1 / (3 * term + 1e-6)

        dZ3 = torch.where(
            deltanet >= 0,
            2 * dZ2 * ds * deltanet,
            0.
        ) # b,h,l,l

        outputs_dK_1.append(
            2 * dZ3.matmul(
                k[
                    ..., 
                    j*chunk_size:(j+1)*chunk_size,
                    :
                ]
            )
        ) # col chunks
        outputs_dK_2.append(
            dY.transpose(-2, -1).matmul(q)
        ) # row chunks

        # dest and src
        outputs_dsrc.append(
            (dZ2 * dest.unsqueeze(-1) * deltanet_relu2).sum(-2)
        ) # b,h,c

        outputs_ddest.append(
            (
                dZ2 * 
                src[
                    ..., 
                    j*chunk_size:(j+1)*chunk_size,
                ].unsqueeze(-2) * 
                deltanet_relu2
            ).sum(-1) # b,h,l
        )

    return (
        (
            sum(outputs_dK_1)
            + torch.cat(outputs_dK_2, dim=-2)
        ),
        torch.cat(outputs_dV, dim=-2),
        torch.cat(outputs_dQ, dim=-2),
        torch.cat(outputs_dsrc, dim=-1),
        sum(outputs_ddest),
    )

class UniversalAttention(Function):

    @staticmethod
    def forward(
        ctx, k, v, q, src, dest,
    ):
        # Assume k is normalized and 
        # src and dest already sigmoided

        # Pass1: get the chunked kernel
        decay_chunks = chunked_decay(
            k, src, dest, 
            skip_preprocessing=True,
        )

        # - we then need to pass the chunks forward
        # NOTE: maybe write a kernel
        decay_chunks = decay_chunks.cumsum(-2)

        # Pass2: run the softmax
        o, score_denom = softmax_with_decay_fwd(
            q, k, v, 
            src, dest, 
            decay_chunks,
            return_denom=True,
            return_decay=False,
            skip_preprocessing=True,
        )

        ctx.save_for_backward(
            k, v, q, src, dest, decay_chunks,
            score_denom,
        )

        return o

    @staticmethod
    def backward(ctx, dout):
        # Note: when using mixed precision, dout is downcast but ddenom is always fp32

        (
            k, v, q, src, dest, decay_chunks,
            score_denom, 
        ) = ctx.saved_tensors

        # NOTE: the below is equivalen to the following drafts
        # dK, dV, dQ, dsrc, ddest = _backward_slow_draft(
        #     dout, 
        #     k, v, q, 
        #     src.sigmoid(), dest.sigmoid(),
        #     decay_chunks,
        #     score_denom,
        # )

        dQ, dK1, dZScoreSum, dZ1_chunked, ddest = rowwise_bwd(
            dout, q, k, v, src, dest,
            decay_chunks, 
            score_denom,
            skip_preprocessing=True,
        )

        dK2, dV, dsrc = colwise_bwd(
            dout, q, k, v, 
            src, dest,
            score_denom,
            dZScoreSum,
            dZ1_chunked,
            skip_preprocessing=True,
        )

        # NOTE: this is needed only if we allow
        # src and dest before the sigmoid
        # we need to accomodate for the transformation, since the above
        # derivation assumed src and dest were sigmoided
        # dsrc *= (
        #     src.sigmoid() * (1 - src.sigmoid())
        # )
        # ddest *= (
        #     dest.sigmoid() * (1 - dest.sigmoid())
        # )

        return (
            dK1 + dK2, # k
            dV, 
            dQ, 
            dsrc, 
            ddest,
        )