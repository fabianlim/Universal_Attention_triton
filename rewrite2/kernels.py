import torch
import triton
import triton.language as tl
from itertools import product

CHUNK_SIZE = 16

CONFIGS = [
    triton.Config({
            'BLOCK_C': c, 
            'BLOCK_D': d,
        }, 
        num_stages=s, num_warps=w
    )
    for c, d, s, w in product(
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
    chunk_size: int = CHUNK_SIZE,
    return_denom: bool = False,
):
    if chunk_size is None:
        chunk_size = CHUNK_SIZE

    # TODO: check sizes
    b, nheads, qlen, qdim = q.shape
    _, kvheads, klen, _ = k.shape
    _, _, _, vdim = v.shape
    assert qlen == klen
    assert nheads % kvheads == 0
    assert qdim == vdim

    num_chunks = triton.cdiv(klen, chunk_size)
    grid = (b, nheads, num_chunks)

    res = torch.zeros(
        (b, nheads, qlen, vdim), 
        device=q.device, 
        dtype=torch.float32
    ) 
    res_attn = torch.zeros(
        (b, nheads, qlen, klen), 
        device=q.device, 
        dtype=torch.float32
    ) 

    # need to use this to renorm the attn later
    res_denom = torch.zeros(
        (
            b, nheads,  
            qlen,
            num_chunks,
        ), 
        device=q.device, 
        dtype=torch.float32
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
        chunk_size=chunk_size,
    )

    if return_denom:
        return res, res_attn, res_denom

    # NOTE: this will cause problems if chunk_size
    # does not divide qlen
    assert qlen % chunk_size == 0
    # NOTE: plan to write this into a kernel
    # normalize each row chunk's attn weights
    for c in range(num_chunks):
        denom = res_denom[
            ...,
            c*chunk_size:(c+1)*chunk_size,
            :(c+1)
        ] # b, h, chunk, c * chunks
        denom -= denom[...,c:(c+1)] # norm by the final one
        res_attn[
            ..., 
            c*chunk_size:(c+1)*chunk_size,
            :(c+1)*chunk_size
        ] *= torch.repeat_interleave(
            torch.exp(denom), # denom is in log
            chunk_size, dim=-1
        ) # re-normalize

    return res, res_attn

@triton.autotune(
    CONFIGS,
    key=['BLOCK_C', 'BLOCK_D'],
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
        res_attn_r += res_attn_stride_kseq * chunk_size
        res_denom_r += res_attn_stride_chunk

        # handle the limit
        limit_c -= BLOCK_C

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