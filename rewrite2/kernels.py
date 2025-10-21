import torch
import triton
import triton.language as tl
from itertools import product

SMALLEST_BLOCK_C = 16
CONFIGS = [
    triton.Config({
            'chunk_size': chunk,
            'BLOCK_C': c, 
            'BLOCK_D': d,
        }, 
        num_stages=s, num_warps=w
    )
    for chunk, c, d, s, w in product(
        [16],
        [SMALLEST_BLOCK_C],
        [16],
        [0],
        [1],
    )
]

CONFIGS_COL = [
    triton.Config({
            'chunk_size': chunk,
            'BLOCK_R': r, 
        }, 
        num_stages=s, num_warps=w
    )
    for chunk, r, s, w in product(
        [16],
        [16],
        [0],
        [1],
    )
]

def softmax_with_decay_fwd(
    q: torch.Tensor, # b,h,l,d
    k: torch.Tensor, # b,h,l,d
    v: torch.Tensor, # b,h,l,d
    decay: torch.Tensor,
    smallest_block_c: int = SMALLEST_BLOCK_C,
    return_denom: bool = False,
):
    # TODO: check sizes
    b, nheads, qlen, qdim = q.shape
    _, kvheads, klen, _ = k.shape
    _, _, _, vdim = v.shape
    assert qlen == klen
    assert nheads % kvheads == 0
    assert qdim == vdim

    grid = lambda META: (b, nheads, triton.cdiv(klen, META['chunk_size']))

    res = torch.zeros(
        (b, nheads, qlen, vdim), 
        device=q.device, 
        dtype=q.dtype,
    ) 
    res_attn = torch.zeros(
        (b, nheads, qlen, klen), 
        device=q.device, 
        dtype=q.dtype,
    ) 

    # NOTE: dont really have a good solution
    # for this. 
    # - dont really need that this much memory
    # - but its troublesome to set the BLOCK_C
    #   value here, 
    res_denom = torch.zeros(
        (
            b, nheads,  
            qlen,
            klen // smallest_block_c
        ), 
        device=q.device, 
        dtype=q.dtype
    ) 

    _softmax_with_decay_fwd[grid](
        res, 
        res_attn,
        res_denom,
        q, k, v, 
        decay,
        res.stride(0), res.stride(1), res.stride(2), res.stride(3),
        res_attn.stride(0), res_attn.stride(1), res_attn.stride(2), res_attn.stride(3),
        res_denom.stride(0), res_denom.stride(1), res_denom.stride(2), res_denom.stride(3),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        decay.stride(0), decay.stride(1), decay.stride(2), decay.stride(3),
        seqlen=klen,
        HEAD_DIM=qdim,
        group_size=nheads // kvheads,
    )

    if return_denom:
        return res, res_attn, res_denom
    return res, res_attn

@triton.autotune(
    CONFIGS,
    key=['chunk_size', 'BLOCK_C', 'BLOCK_D'],
)
@triton.jit
def _softmax_with_decay_fwd(
    res, 
    res_attn,
    res_denom,
    queries, keys, values, 
    decay, 
    res_stride_b, res_stride_h, res_stride_seq, res_stride_dim,
    res_attn_stride_b, res_attn_stride_h, res_attn_stride_qseq,  res_attn_stride_kseq,
    res_denom_stride_b, res_denom_stride_h, res_denom_stride_qseq,  res_attn_stride_chunk,
    q_stride_b, q_stride_h, q_stride_seq, q_stride_dim,
    k_stride_b, k_stride_h, k_stride_seq, k_stride_dim,
    v_stride_b, v_stride_h, v_stride_seq, v_stride_dim,
    d_stride_b, d_stride_h, d_stride_qseq, d_stride_kseq,
    seqlen: int,
    group_size: int,
    chunk_size: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_b = tl.program_id(0) # batch
    pid_h = tl.program_id(1) # query head
    hkv = pid_h // group_size # key / value head
    pid_r = tl.program_id(2) # row chunk

    # offset by batch and head
    res += pid_b * res_stride_b + pid_h * res_stride_h
    res_attn += pid_b * res_attn_stride_b + pid_h * res_attn_stride_h
    res_denom += pid_b * res_denom_stride_b + pid_h * res_denom_stride_h
    queries += pid_b * q_stride_b + pid_h * q_stride_h
    keys += pid_b * k_stride_b + hkv * k_stride_h
    values += pid_b * v_stride_b + hkv * v_stride_h
    decay += pid_b * d_stride_b + hkv * d_stride_h

    # keys (row) and dest will be offset by chunk
    # keys_r = keys + pid_r * chunk_size * k_stride_seq
    queries_r = queries + pid_r * chunk_size * q_stride_seq
    decay_r = decay + pid_r * chunk_size * d_stride_qseq
    res_r = res + pid_r * chunk_size * res_stride_seq
    res_attn_r = res_attn + pid_r * chunk_size * res_attn_stride_qseq
    res_denom_r = res_denom + pid_r * chunk_size * res_denom_stride_qseq

    # we allow row chunk size to differ from column chunk,
    # so we 
    # - a the pid_r-th row chunk will take up (pid_r * chunk_size) columns
    # - so 
    nC = tl.cdiv(
        (pid_r + 1) * chunk_size - 1, BLOCK_C
    ) # number of column chunks to process (including the spillover)
    nD = tl.cdiv(HEAD_DIM, BLOCK_D)

    # load scales
    # qk_scale = 1.44269504  # 1/log(2)
    offs_i = tl.arange(0, chunk_size)
    offs_j = tl.arange(0, BLOCK_C) # columns
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = tl.arange(0, HEAD_DIM)
    limit_r = seqlen - pid_r * chunk_size # row limit
    limit_c = seqlen # col_limit

    # online-softmax accumulation elements
    # - for those out of row chunk set it 0
    score_max = tl.where( 
        offs_i < limit_r, -float("inf"), 0.0
    )
    score_denom = tl.zeros([chunk_size], dtype=tl.float32) + 1.0
    # TODO: its not gauranteed that value dim 
    # equals to query and key dim
    acc = tl.zeros([chunk_size, HEAD_DIM], dtype=tl.float32)

    # process the columns
    for c in range(0, nC):

        score = tl.zeros((chunk_size, BLOCK_C), dtype=tl.float32)

        # compute the keys (tiled on dimension)
        limit_d = HEAD_DIM # dims
        q_mat_ptr = queries_r
        kt_mat_ptr = keys

        for _ in range(0, nD):

            q_mat = tl.load(
                (
                    q_mat_ptr 
                    + offs_i[:, None] * q_stride_seq
                    + offs_d[None, :] * q_stride_dim
                ),
                mask=(offs_i[:, None] < limit_r) & (offs_d[None, :] < limit_d), 
                other=0.0
            ).to(tl.float32)

            kt_mat = tl.load(
                (
                    kt_mat_ptr 
                    + offs_d[:, None] * k_stride_dim
                    + offs_j[None, :] * k_stride_seq
                ),
                mask= (offs_d[:, None] < limit_d) & (offs_j[None, :] < limit_c),
                other=0.0
            ).to(tl.float32)

            score += tl.dot(q_mat, kt_mat)

            # handle the limit
            limit_d -= BLOCK_D

            # handle the pointers
            q_mat_ptr += BLOCK_D * q_stride_dim
            kt_mat_ptr += BLOCK_D * k_stride_dim

        # load the decay
        decay_mat = tl.load(
            decay_r 
            + offs_i[:, None] * d_stride_qseq
            + offs_j[None, :] * d_stride_kseq,
            mask=(offs_i[:, None] < limit_r) & (offs_j[None, :] < limit_c), 
            other=0.0,
        ).to(tl.float32)

        # ---------- ONLINE SOFTMAX (FLASH ATTENTION) -------------
        score += decay_mat

        # - need to do this for the last chunk ends
        # - so that the below tl.max(score) will not 
        #   take those values into account
        #  x x x | -inf -inf 
        #  x x x | -inf -inf 
        #  0 0 0 |  0    0 
        if limit_c < BLOCK_C:
            score = tl.where(
                offs_j[None, :] < limit_c,
                score,
                - float("inf")
            )

        if limit_r < chunk_size:
            score = tl.where(
                offs_i[:, None] < limit_r,
                score, 0.0
            )

        # Stabilize logsumexp using the subtract max trick
        # - recall above if for off_i >= limit_r we set
        #   score_max - 0.
        # - so score_max = max(score_max, tl.max(score))) = 0 for these rows
        score_max_prev = score_max # m_{i-1}
        score_denom_prev = score_denom # d_{i-1}
        score_max = tl.maximum(
            score_max, 
            tl.max(score, axis=1), 
        ) # m_i
        score_denom_corrected = (
            score_denom_prev * 
            tl.exp(
                score_max_prev - score_max
            ) 
        ) # d_{i-1} * exp(m_{i-1} - m_i)
        weights = tl.exp(
            score - score_max[:, None]
        ) # exp(q^T k - m_i)


        # - similarly, we handle this boundary 
        #   so as to not participate in the tl.sum below
        if limit_c < BLOCK_C:
            weights = tl.where(
                offs_j[None, :] < limit_c,
                weights, 0.0
            )

        # - update score denom
        # d_i = d_{i-1} * exp(m_{i-1} - m_i) + \sum_{j} exp(q^T k - m_i)
        score_denom = (
            score_denom_corrected + tl.sum(weights, axis=1)
        )

        v_mat = tl.load(
            (
                values
                + offs_j[:, None] * v_stride_seq
                + offs_v[None, :] * v_stride_dim
            ),
            mask=(offs_j[:, None] < limit_c),
            other=0.0
        ).to(tl.float32)

        # o_{i-1} * d_{i-1} * exp(m_{i-1} - m_i) / d_i
        # +  \sum_{j} exp(q^T k - m_i) / d_i *  V[j]
        acc *= (score_denom_corrected / score_denom)[:, None]
        acc += tl.dot(
            weights / score_denom[:, None],
            v_mat
        )

        # this is the equiv norm without the 
        # max normalization
        # NOTE: the last store is not useful
        tl.store(
            res_denom_r + offs_i * res_denom_stride_qseq,
            score_max + tl.log(score_denom),
            mask=offs_i < limit_r
        )

        tl.store(
            (
                res_attn_r
                + offs_i[:, None] * res_attn_stride_qseq
                + offs_j[None, :] * res_attn_stride_kseq
            ),
            weights / score_denom[:, None],
            mask=(
                (offs_i[:, None] < limit_r) &
                (offs_j[None, :] < limit_c)
            )
        )

        # move pointer with column chunk
        keys += BLOCK_C * k_stride_seq
        values += BLOCK_C * v_stride_seq
        res_attn_r += BLOCK_C * res_attn_stride_kseq
        res_denom_r += res_attn_stride_chunk
        decay_r += BLOCK_C * d_stride_kseq

        # handle the limit
        limit_c -= BLOCK_C

    # go through again (backwards) and update the 
    # attn weights
    score_denom_corrected = score_max + tl.log(score_denom) 
    for c in range(0, nC):

        res_attn_r -= BLOCK_C * res_attn_stride_kseq
        res_denom_r -= res_attn_stride_chunk

        # handle the limit
        limit_c += BLOCK_C

        if c > 0:
            attn_mat = tl.load(
                (
                    res_attn_r 
                    + offs_i[:, None] * res_attn_stride_qseq
                    + offs_j[None, :] * res_attn_stride_kseq
                ),
                mask=(
                    (offs_i[:, None] < limit_r) &
                    (offs_j[None, :] < limit_c)
                ),
                other=0.0
            ).to(tl.float32)

            old_denom = tl.load(
                res_denom_r + offs_i * res_denom_stride_qseq,
                mask=offs_i < limit_r,
                other=0.0
            )

            # renormalization factor
            old_denom -= score_denom_corrected

            tl.store(
                (
                    res_attn_r
                    + offs_i[:, None] * res_attn_stride_qseq
                    + offs_j[None, :] * res_attn_stride_kseq
                ),
                attn_mat * tl.exp(old_denom)[:, None],
                mask=(
                    (offs_i[:, None] < limit_r) &
                    (offs_j[None, :] < limit_c)
                )
            )

    # -  DONE WITH COL CHUNK LOOPS - 

    tl.store(
        (
            res_r 
            + offs_i[:, None] * res_stride_seq
            + offs_v[None, :] * res_stride_dim
        ),
        acc, 
        mask=(
            (offs_i[:, None] < limit_r)
        )
    )

def compute_dYdQ(
    dout: torch.Tensor, # b,h,l,d
    q: torch.Tensor, # b,h,l,d
    k: torch.Tensor, # b,h,l,d
    v: torch.Tensor, # b,h,l,d
    attn: torch.Tensor,
):

    # TODO: check sizes
    b, nheads, qlen, qdim = q.shape
    _, kvheads, klen, kdim = k.shape
    _, _, _, vdim = v.shape
    assert qlen == klen
    assert nheads % kvheads == 0
    assert qdim == vdim

    grid = lambda META: (b, nheads, triton.cdiv(klen, META['chunk_size']))

    res_dY = torch.zeros(
        (b, nheads, qlen, klen), 
        device=q.device, 
        dtype=q.dtype,
    ) 

    res_dQ = torch.zeros(
        (b, nheads, qlen, qdim), 
        device=q.device, 
        dtype=q.dtype
    ) 

    _compute_dYdQ[grid](
        res_dY, 
        res_dQ,
        dout,
        attn,
        v, k,
        res_dY.stride(0), res_dY.stride(1), res_dY.stride(2), res_dY.stride(3),
        res_dQ.stride(0), res_dQ.stride(1), res_dQ.stride(2), res_dQ.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        attn.stride(0), attn.stride(1), attn.stride(2), attn.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        seqlen=klen,
        HEAD_DIM=qdim,
        group_size=nheads // kvheads,
    )

    return res_dY, res_dQ


@triton.autotune(
    CONFIGS,
    key=['chunk_size', 'BLOCK_C', 'BLOCK_D'],
)
@triton.jit
def _compute_dYdQ(
    res_dY, 
    res_dQ, 
    dout, 
    attn,
    values, keys,
    res_dY_stride_b, res_dY_stride_h, res_dY_stride_qseq, res_dY_stride_kseq,
    res_dQ_stride_b, res_dQ_stride_h, res_dQ_stride_qseq, res_dQ_stride_dim,
    do_stride_b, do_stride_h, do_stride_seq, do_stride_dim,
    attn_stride_b, attn_stride_h, attn_stride_qseq,  attn_stride_kseq,
    v_stride_b, v_stride_h, v_stride_seq, v_stride_dim,
    k_stride_b, k_stride_h, k_stride_seq, k_stride_dim,
    seqlen: int,
    group_size: int,
    chunk_size: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):

    pid_b = tl.program_id(0) # batch
    pid_h = tl.program_id(1) # query head
    hkv = pid_h // group_size # key / value head
    pid_r = tl.program_id(2) # row chunk

    # offset by batch and head
    res_dY += pid_b * res_dY_stride_b + pid_h * res_dY_stride_h
    res_dQ += pid_b * res_dQ_stride_b + pid_h * res_dQ_stride_h
    dout += pid_b * do_stride_b + pid_h * do_stride_h
    attn += pid_b * attn_stride_b + pid_h * attn_stride_h
    values += pid_b * v_stride_b + hkv * v_stride_h
    keys += pid_b * k_stride_b + hkv * k_stride_h

    # rows
    res_dY_r = res_dY + pid_r * chunk_size * res_dY_stride_qseq
    res_dQ_r = res_dQ + pid_r * chunk_size * res_dQ_stride_qseq
    attn_r = attn + pid_r * chunk_size * attn_stride_qseq
    dout_r = dout + pid_r * chunk_size * do_stride_seq

    # we allow row chunk size to differ from column chunk,
    # so we 
    # - a the pid_r-th row chunk will take up (pid_r * chunk_size) columns
    # - so 
    nC = tl.cdiv(
        (pid_r + 1) * chunk_size - 1, BLOCK_C
    ) # number of column chunks to process (including the spillover)
    nD = tl.cdiv(HEAD_DIM, BLOCK_D)

    offs_i = tl.arange(0, chunk_size)
    offs_j = tl.arange(0, BLOCK_C) # columns
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = tl.arange(0, HEAD_DIM)
    limit_r = seqlen - pid_r * chunk_size # row limit
    limit_c = seqlen # col_limit

    # in the first pass we compute dZScore
    dZScore_sum = tl.zeros([chunk_size], dtype=tl.float32)

    # must define here for compilation purposes
    dZ = tl.zeros((chunk_size, BLOCK_C), dtype=tl.float32)
    attn_mat = tl.load(
        (
            attn_r 
            + offs_i[:, None] * attn_stride_qseq
            + offs_j[None, :] * attn_stride_kseq
        ),
        mask=(
            (offs_i[:, None] < limit_r) &
            (offs_j[None, :] < limit_c)
        ),
        other=0.0
    ).to(tl.float32)

    # process the columns
    for c in range(0, nC):

        dZ = tl.zeros((chunk_size, BLOCK_C), dtype=tl.float32)

        do_mat_ptr = dout_r
        vt_mat_ptr = values

        # ------- dZ,dZScore -------
        limit_d = HEAD_DIM # dims
        for _ in range(0, nD):

            # for dz
            # NOTE: we assme now dim(do) = dim(v) = HEAD_DIM
            do_mat = tl.load(
                (
                    do_mat_ptr 
                    + offs_i[:, None] * do_stride_seq
                    + offs_d[None, :] * do_stride_dim
                ),
                mask=(offs_i[:, None] < limit_r) & (offs_d[None, :] < limit_d), 
                other=0.0
            ).to(tl.float32)

            # for dZ
            vt_mat = tl.load(
                (
                    vt_mat_ptr 
                    + offs_d[:, None] * v_stride_dim
                    + offs_j[None, :] * v_stride_seq
                ),
                mask= (offs_d[:, None] < limit_d) & (offs_j[None, :] < limit_c),
                other=0.0
            ).to(tl.float32)

            dZ += tl.dot(do_mat, vt_mat)

            # handle the limit
            limit_d -= BLOCK_D

            # handle the pointers
            do_mat_ptr += BLOCK_D * do_stride_dim
            vt_mat_ptr += BLOCK_D * v_stride_dim

        # dZ * Z
        dZ *= attn_mat

        # store dZ *Z here first
        tl.store(
            (
                res_dY_r
                + offs_i[:, None] * res_dY_stride_qseq
                + offs_j[None, :] * res_dY_stride_kseq
            ),
            dZ,
            mask=(
                (offs_i[:, None] < limit_r) &
                (offs_j[None, :] < limit_c)
            )
        )

        # compute (dZ * Z).sum(-1)
        dZScore_sum += tl.sum(dZ, axis=1)

        # move pointer with column chunk
        values += BLOCK_C * v_stride_seq
        attn_r += BLOCK_C * attn_stride_kseq
        res_dY_r += BLOCK_C * res_dY_stride_kseq

        # handle the limit
        limit_c -= BLOCK_C

        # for the next iteration
        if c < (nC-1):
            attn_mat = tl.load(
                (
                attn_r 
                    + offs_i[:, None] * attn_stride_qseq
                    + offs_j[None, :] * attn_stride_kseq
                ),
                mask=(
                    (offs_i[:, None] < limit_r) &
                    (offs_j[None, :] < limit_c)
                ),
                other=0.0
            ).to(tl.float32)

    # for dQ
    acc = tl.zeros([chunk_size, HEAD_DIM], dtype=tl.float32)
    keys += nC * BLOCK_C * k_stride_seq

    # go through again (backwards) and update the 
    # - dZZ - dZScore_sum * Z
    for c in range(0, nC):

        # move pointer (backward) with column chunk
        attn_r -= BLOCK_C * attn_stride_kseq
        res_dY_r -= BLOCK_C * res_dY_stride_kseq
        keys -= BLOCK_C * k_stride_seq

        # handle the limit
        limit_c += BLOCK_C

        if c > 0:
            attn_mat = tl.load(
                (
                    attn_r 
                    + offs_i[:, None] * attn_stride_qseq
                    + offs_j[None, :] * attn_stride_kseq
                ),
                mask=(
                    (offs_i[:, None] < limit_r) &
                    (offs_j[None, :] < limit_c)
                ),
                other=0.0
            ).to(tl.float32)

            # remember dZZ was saved here
            dZ = tl.load(
                (
                    res_dY_r 
                    + offs_i[:, None] * attn_stride_qseq
                    + offs_j[None, :] * attn_stride_kseq
                ),
                mask=(
                    (offs_i[:, None] < limit_r) &
                    (offs_j[None, :] < limit_c)
                ),
                other=0.0
            ).to(tl.float32)

        # dY
        dY = dZ - dZScore_sum[:, None] * attn_mat

        # dQ
        k_mat = tl.load(
            (
               keys 
                + offs_j[:, None] * k_stride_seq
                + offs_v[None, :] * k_stride_dim # NOTE: assumed same
            ),
            mask=(offs_j[:, None] < limit_c),
            other=0.0
        ).to(tl.float32)
        acc += tl.dot(dY, k_mat)

        # update the result
        tl.store(
            (
                res_dY_r
                + offs_i[:, None] * res_dY_stride_qseq
                + offs_j[None, :] * res_dY_stride_kseq
            ),
            dY,
            mask=(
                (offs_i[:, None] < limit_r) &
                (offs_j[None, :] < limit_c)
            )
        )

    # update dQ
    tl.store(
        (
            res_dQ_r 
            + offs_i[:, None] * res_dQ_stride_qseq
            + offs_v[None, :] * res_dQ_stride_dim
        ),
        acc,
        mask=(
            (offs_i[:, None] < limit_r)
        )
    )

def compute_dVdK(
    dout: torch.Tensor, # b,h,l,d
    q: torch.Tensor, # b,h,l,d
    dY: torch.Tensor, # b,h,l,l
    attn: torch.Tensor, # b,h,l,l
    kvheads: int,
):

    # TODO: check sizes
    b, nheads, qlen, qdim = q.shape
    _, _, dolen, vdim = dout.shape

    grid = lambda META: (b, kvheads, triton.cdiv(dolen, META['chunk_size']))

    res_dV = torch.zeros(
        (b, kvheads, dolen, vdim), 
        device=q.device, 
        dtype=q.dtype,
    ) 

    res_dK = torch.zeros(
        (b, kvheads, dolen, qdim), # assume same
        device=q.device, 
        dtype=q.dtype,
    ) 

    _compute_dVdK[grid](
        res_dV, res_dK,
        dout, dY, attn,
        q,
        res_dV.stride(0), res_dV.stride(1), res_dV.stride(2), res_dV.stride(3),
        res_dK.stride(0), res_dK.stride(1), res_dK.stride(2), res_dK.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        dY.stride(0), dY.stride(1), dY.stride(2), dY.stride(3),
        attn.stride(0), attn.stride(1), attn.stride(2), attn.stride(3),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        seqlen=qlen,
        HEAD_DIM=vdim,
        group_size=nheads // kvheads,
    )

    # NOTE: need to collapse heads later

    return res_dV, res_dK

@triton.autotune(
    CONFIGS_COL,
    key=['BLOCK_R'],
)
@triton.jit
def _compute_dVdK(
    res_dV,
    res_dK,
    dout, 
    dY, 
    attn,
    queries,
    res_dV_stride_b, res_dV_stride_h, res_dV_stride_seq, res_dV_stride_dim,
    res_dK_stride_b, res_dK_stride_h, res_dK_stride_seq, res_dK_stride_dim,
    do_stride_b, do_stride_h, do_stride_seq, do_stride_dim,
    dY_stride_b, dY_stride_h, dY_stride_qseq, dY_stride_kseq,
    attn_stride_b, attn_stride_h, attn_stride_qseq,  attn_stride_kseq,
    q_stride_b, q_stride_h, q_stride_seq, q_stride_dim,
    seqlen: int,
    group_size: int,
    chunk_size: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_b = tl.program_id(0) # batch
    hkv = tl.program_id(1) # kv head
    pid_c = tl.program_id(2) # col chunk

    # offset by batch and head
    dout += pid_b * do_stride_b + hkv * group_size * do_stride_h
    dY += pid_b * dY_stride_b + hkv * group_size * dY_stride_h
    attn += pid_b * attn_stride_b + hkv * group_size * attn_stride_h
    queries += pid_b * q_stride_b + hkv * group_size * q_stride_h

    # - dep on kvhead
    res_dV += pid_b * res_dV_stride_b + hkv * res_dV_stride_h
    res_dK += pid_b * res_dK_stride_b + hkv * res_dK_stride_h

    # columns
    dout_r = dout + pid_c * chunk_size * do_stride_seq
    queries_r = queries + pid_c * chunk_size * q_stride_seq
    res_dV_c = res_dV + pid_c * chunk_size * res_dV_stride_seq
    res_dK_c = res_dK + pid_c * chunk_size * res_dK_stride_seq

    # - start in the correct block
    dY_c = dY + pid_c * chunk_size * (dY_stride_qseq + dY_stride_kseq)
    attn_c = attn + pid_c * chunk_size * (attn_stride_qseq + attn_stride_kseq)

    # we allow row chunk size to differ from column chunk,
    # so we 
    # - a the pid_r-th row chunk will take up (pid_r * chunk_size) columns
    # - so 
    nR = tl.cdiv(
        seqlen - pid_c * chunk_size, BLOCK_R
    ) # number of column chunks to process (including the spillover)

    # load scales
    offs_i = tl.arange(0, BLOCK_R) # rows
    offs_j = tl.arange(0, chunk_size)
    offs_v = tl.arange(0, HEAD_DIM)
    limit_r = seqlen # row_limit
    limit_c = seqlen - pid_c * chunk_size # col limit

    # computation of dK_2
    acc = tl.zeros([HEAD_DIM, chunk_size], dtype=tl.float32)

    # computation of dV
    acc2 = tl.zeros([HEAD_DIM, chunk_size], dtype=tl.float32)

    # process the rows
    for r in range(0, nR):

        for i in range(0, group_size):

            dot_mat = tl.load(
                (
                    dout_r 
                    + i * do_stride_h
                    + offs_v[:, None] * do_stride_dim # NOTE: assumed same
                    + offs_i[None, :] * do_stride_seq
                ),
                mask=(offs_i[None, :] < limit_r),
                other=0.0
            ).to(tl.float32)

            attn_mat = tl.load(
                (
                    attn_c 
                    + i * attn_stride_h
                    + offs_i[:, None] * attn_stride_qseq
                    + offs_j[None, :] * attn_stride_kseq
                ),
                mask=(
                    (offs_i[:, None] < limit_r) &
                    (offs_j[None, :] < limit_c)
                ),
                other=0.0
            ).to(tl.float32)

            # dV
            # - score should have zeros in appropriate places
            acc2 += tl.dot(dot_mat, attn_mat)

            qt_mat = tl.load(
                (
                    queries_r
                    + i * q_stride_h
                    + offs_v[:, None] * q_stride_dim # NOTE: assumed same
                    + offs_i[None, :] * q_stride_seq
                ),
                mask=(offs_i[None, :] < limit_r),
                other=0.0
            ).to(tl.float32)

            dY_mat = tl.load(
                (
                    dY_c 
                    + i * dY_stride_h
                    + offs_i[:, None] * dY_stride_qseq
                    + offs_j[None, :] * dY_stride_kseq
                ),
                mask=(
                    (offs_i[:, None] < limit_r) &
                    (offs_j[None, :] < limit_c)
                ),
                other=0.0
            ).to(tl.float32)

            # dK
            # - score (and therefore dZScore) should have
            #   zeros in appropriate places
            acc += tl.dot(qt_mat, dY_mat)

        # movements
        queries_r += BLOCK_R * q_stride_seq
        dY_c += BLOCK_R * dY_stride_qseq
        attn_c += BLOCK_R * attn_stride_qseq
        dout_r += BLOCK_R * do_stride_seq

        # handle the limit
        limit_r -= BLOCK_R

    # -  DONE WITH ROW CHUNK LOOPS - 

    tl.store(
        (
            res_dV_c 
            + offs_v[:, None] * res_dV_stride_dim
            + offs_j[None, :] * res_dV_stride_seq
        ),
        acc2,
        mask=(
            (offs_j[None, :] < limit_c)
        )
    )

    tl.store(
        (
            res_dK_c 
            + offs_v[:, None] * res_dK_stride_dim
            + offs_j[None, :] * res_dK_stride_seq
        ),
        acc,
        mask=(
            (offs_j[None, :] < limit_c)
        )
    )