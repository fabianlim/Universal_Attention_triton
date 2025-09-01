import torch
import triton
import triton.language as tl

def ua_forward(
    q: torch.Tensor, # b,h,l,d
    k: torch.Tensor, # b,h,l,d
    v: torch.Tensor, # b,h,l,d
    static_src: torch.Tensor,
    static_dest: torch.Tensor,
    chunk_size: int = 16,
    return_decay: bool = False,
):
    # TODO: check sizes
    b, nheads, qlen, qdim = q.shape
    _, kvheads, klen, _ = k.shape
    _, _, _, vdim = v.shape
    assert qlen == klen
    assert nheads % kvheads == 0

def chunked_decay(
    keys: torch.Tensor, # b,h,l,d
    src: torch.Tensor,
    dest: torch.Tensor,
    chunk_size: int = 16,
):

    b, kvheads, klen, kdim = keys.shape
    num_chunks = triton.cdiv(klen, chunk_size)
    res = torch.zeros(
        (b, kvheads, num_chunks, klen), 
        device=keys.device, 
        #dtype=torch.float32
    )

    grid = (b, kvheads, num_chunks)

    # NOTE: move this somewhere?
    # L2-normalize K
    keys = keys / keys.pow(2).sum(-1,True).sqrt().add(1e-6)

    # sigmoid
    src = src.sigmoid()
    dest = dest.sigmoid()

    # NOTE: 
    # - static_src and static_dest assumed to be sigmoided

    _chunked_decay[grid](
        res, keys, src, dest,
        res.stride(0), res.stride(1), res.stride(2), res.stride(3),
        keys.stride(0), keys.stride(1), keys.stride(2), keys.stride(3),
        src.stride(0), src.stride(1), src.stride(2),
        dest.stride(0), dest.stride(1), dest.stride(2),
        seqlen=klen,
        HEAD_DIM=kdim,
        chunk_size=chunk_size,
    )

    return res

# inspired by Universal_Attention.triton.universal_attention_kernel._universal_attention_fwd_kernel

@triton.autotune(
    [
        triton.Config({'BLOCK_C': 16, 'BLOCK_D': 16}, num_stages=1, num_warps=1),
    ],
    key=['BLOCK_C', 'BLOCK_D'],
)
@triton.jit
def _chunked_decay(
    res, keys, src, dest,
    res_stride_b, res_stride_h, res_stride_chunk, res_stride_kseq,
    k_stride_b, k_stride_h, k_stride_seq, k_stride_dim,
    src_stride_b, src_stride_h, src_stride_seq, 
    dest_stride_b, dest_stride_h, dest_stride_seq, 
    seqlen: int,
    chunk_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_b = tl.program_id(0) # batch
    pid_h = tl.program_id(1) # query head
    pid_r = tl.program_id(2) # row chunk

    # offset by batch and head
    res += pid_b * res_stride_b + pid_h * res_stride_h
    keys += pid_b * k_stride_b + pid_h * k_stride_h
    src += pid_b * src_stride_b + pid_h * src_stride_h
    dest += pid_b * dest_stride_b + pid_h * dest_stride_h

    # keys (row) and dest will be offset by chunk
    keys_r = keys + pid_r * chunk_size * k_stride_seq
    dest += pid_r * chunk_size * dest_stride_seq

    # res offset by chunk index
    res += pid_r * res_stride_chunk

    # we allow row chunk size to differ from column chunk,
    # so we 
    # - a the pid_r-th row chunk will take up (pid_r * chunk_size) columns
    # - so 
    nC = tl.cdiv(
        (pid_r + 1) * chunk_size - 1, BLOCK_C
    ) # number of column chunks to process (including the spillover)
    nD = tl.cdiv(HEAD_DIM, BLOCK_D)

    offs_i = tl.arange(0, chunk_size) # rows
    offs_j = tl.arange(0, BLOCK_C) # columns
    offs_d = tl.arange(0, BLOCK_D) # dim
    limit_r = seqlen - pid_r * chunk_size # row limit
    limit_c = seqlen # col_limit

    # load the dest
    dest_vec = tl.load(
        dest + offs_i * dest_stride_seq,
        mask=offs_i < limit_r,
        other=0.0,
    )
    dest_vec = tl.exp2(tl.log2(dest_vec) / 3.0) # pow (1/3)

    # process the columns
    for c in range(0, nC):

        # process tile at a time
        affinity = tl.zeros((chunk_size, BLOCK_C), dtype=tl.float32)

        # compute the keys (tiled on dimension)
        limit_d = HEAD_DIM # dims
        k_mat_ptr = keys_r
        kt_mat_ptr = keys
        for _ in range(0, nD):

            # TODO: for the last block need to do the triangular masking
            # NOTE: this row load can actually be optimized
            # - since its repetitive, especially in the case nD == 1
            k_mat = tl.load(
                (
                    k_mat_ptr + offs_i[:, None] * k_stride_seq
                    + offs_d[None, :] * k_stride_dim
                ),
                mask=(offs_i[:, None] < limit_r) & (offs_d[None, :] < limit_d), 
                other=0.0
            )
            kt_mat = tl.load(
                (
                    kt_mat_ptr 
                    + offs_d[:, None] * k_stride_dim
                    + offs_j[None, :] * k_stride_seq
                ),
                mask=(offs_i[:, None] < limit_c) & (offs_d[None, :] < limit_d), 
                other=0.0
            )

            # TODO: handle precision
            affinity += tl.dot(k_mat, kt_mat, input_precision="ieee")

            # handle the limit
            limit_d -= BLOCK_D

            # handle the pointers
            k_mat_ptr += BLOCK_D * k_stride_dim
            kt_mat_ptr += BLOCK_D * k_stride_dim

        # .relu().pow(2/3)
        affinity = tl.exp2(tl.log2(tl.maximum(affinity, 0.0)) * 2.0 / 3.0)

        # load the src
        src_vec = tl.load(
            src + offs_j * src_stride_seq,
            mask=offs_j < limit_c,
            other=0.0,
        )
        src_vec = tl.exp2(tl.log2(src_vec) / 3.0) # pow (1/3)
        affinity = affinity * dest_vec[:, None] * src_vec[None, :]

        # - convert to log(1-p)
        # torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())
        decay = tl.log(1.0 - tl.clamp(affinity, 0.0, 1.0 - 1e-6)) 

        # need to zero out the upper triangular since these will 
        # provide zero log-likelihoods
        # - only possible if there is some (i,j) such that
        #   i  + pid_r * chunk_size > j + c * BLOCK_C
        # - this is equivalent to 
        #   i > j + offset 
        #   where offset = c * BLOCK_C - pid_r * chunk_size
        # - this is trivally satisfied if offset < 0
        # - so we check chunk_size - 1 > offset
        # - same as chunk_size >= offset

        # if the rightmost element of the block
        # is greater
        offset = c * BLOCK_C - pid_r * chunk_size
        if offset >= 0 and chunk_size >= offset:
            decay = tl.where(
                (offs_i[:, None] > (offs_j[None, :] + offset)), 
                decay, 0.0
            )

        # accumulate over the chunk rows
        decay = tl.sum(decay, axis=-2)

        tl.store(
            res + offs_i, decay,
            mask=offs_i < limit_c
        )

        # move pointer with column chunk
        keys += BLOCK_C * k_stride_seq
        res += BLOCK_C * res_stride_kseq
        src += BLOCK_C * src_stride_seq

        # handle the limit
        limit_c -= BLOCK_C


def softmax_with_decay_fwd(
    q: torch.Tensor, # b,h,l,d
    k: torch.Tensor, # b,h,l,d
    v: torch.Tensor, # b,h,l,d
    static_src: torch.Tensor,
    static_dest: torch.Tensor,
    chunk_size: int = 16,
    return_decay: bool = False,
):
    # TODO: check sizes
    b, nheads, qlen, qdim = q.shape
    _, kvheads, klen, _ = k.shape
    _, _, _, vdim = v.shape
    assert qlen == klen
    assert nheads % kvheads == 0

    # grid = lambda META: (
    #     b, 
    #     nheads, 
    #     triton.cdiv(qlen, META['BLOCK_SIZE_M'])
    # )
    grid = (b, nheads, triton.cdiv(qlen, chunk_size))

    res = torch.empty(
        (b, nheads, qlen, vdim), 
        device=q.device, 
        #dtype=torch.float32
    ) 
    decay = None
    if return_decay:
        decay = torch.empty(
            (b, nheads, qlen, klen), 
            device=q.device, 
            #dtype=torch.float32
        ) 

    _softmax_with_decay_fwd[grid](
        res,
        q, k, v, static_src, static_dest,
        decay,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
    )

triton.jit
def _softmax_with_decay_fwd(
    res, queries, keys, values, 
    static_src, static_dest,
    decay,
    res_stride_b, res_stride_h, res_stride_seq, res_stride_dim,
    q_stride_b, q_stride_h, q_stride_seq, q_stride_dim,
    k_stride_b, k_stride_h, k_stride_seq, k_stride_dim,
    v_stride_b, v_stride_h, v_stride_seq, v_stride_dim,
    d_stride_b, d_stride_h, d_stride_qseq, d_stride_kseq,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    Q_LEN: tl.constexpr,
):
    # dtype = desc_q.dtype.element_ty

    # TODO: need to somehow assert that BLOCK_M divides chunk_size

    pid_b = tl.program_id(0) # batch
    pid_h = tl.program_id(1) # query head
    pid_r = tl.program_id(1) # row chunk

    # move the pointers
    res += pid_b * res_stride_b + pid_h * res_stride_h\
        + pid_r * BLOCK_M * res_stride_seq
    queries += pid_b * q_stride_b + pid_h * q_stride_h\
        + pid_r * BLOCK_M * q_stride_seq
    keys += pid_b * k_stride_b + pid_h * k_stride_h\
        + pid_r * BLOCK_M * k_stride_seq
    values += pid_b * v_stride_b + pid_h * v_stride_h\
        + pid_r * BLOCK_M * v_stride_seq

    # initialize offsets
    # offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # offs_n = tl.arange(0, BLOCK_N)

    # online-softmax accumulation elements
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # load scales
    qk_scale = 1.44269504  # 1/log(2)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # - 


    # ------------ SOFTMAX --------------

    # load the q-chunk
    # q = tl.load(
    #     (
    #         queries + offs_m[:, None] * q_stride_seq 
    #         + offs_d[None, :] * q_stride_dim
    #     ),
    #     mask=(pid_r * BLOCK_M + offs_m[:, None]) < Q_LEN, 
    #     other=0.0
    # )

#     if STAGE & 1:
#         acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
#                                         desc_k, desc_v, desc_affinity,  #
#                                         offset_y, offsetaffinity_y, dtype, start_m, qk_scale,  #
#                                         BLOCK_M, HEAD_DIM, BLOCK_N,  #
#                                         4 - STAGE, offs_m, offs_n, N_CTX, causal, KV_H, Q_H)
# 
#     # epilogue
#     m_i += tl.math.log(l_i)
#     acc = acc / l_i[:, None]
#     m_ptrs = M + off_hz * Q_LEN + offs_m
#     tl.store(m_ptrs, m_i)
#     tl.store(desc_o+qo_ptr, acc, mask=tl.arange(0, BLOCK_M)[:, None]+start_m*BLOCK_M < N_CTX)
# 

@triton.jit
def _softmax_with_decay_fwd(
    acc, l_i, m_i, q,  #
    desc_k, desc_v, desc_affinity, #
    offset_y, offsetaffinity_y, dtype: tl.constexpr, start_m, qk_scale,  #
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr, causal: tl.constexpr, KV_H: tl.constexpr, Q_H: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo * HEAD_DIM
    offsetv_y = offset_y + lo * HEAD_DIM
    offseta_y = offsetaffinity_y + lo 
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k_ptr = offsetk_y + tl.arange(0, BLOCK_N)[:, None]*HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
        aff = tl.load(desc_affinity + offseta_y + tl.arange(0, BLOCK_M)[:, None] * N_CTX + tl.arange(0, BLOCK_N)[None, :], mask=(\
            (tl.arange(0, BLOCK_M)[:, None] + start_m * BLOCK_M < N_CTX) & (tl.arange(0, BLOCK_N)[None, :] + start_n < N_CTX)
        ), other=0.0).to(tl.float32)
        k = tl.trans(tl.load(desc_k + k_ptr, mask=(tl.arange(0, BLOCK_N)+start_n)[:,None] < N_CTX, other=0.0))
        qk = tl.dot(q, k)
        #qk *= 1/tl.sqrt(tl.cast(HEAD_DIM, dtype=tl.float32)) 
        qk += aff
        if STAGE == 2:
            mask = (offs_m[:, None] >= (start_n + offs_n[None, :])) & ((start_n + offs_n[None, :]) < N_CTX) & (offs_m[:, None] < N_CTX)
            qk = qk + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

        p = tl.math.exp(qk)
        # -- compute correction factor
        alpha = tl.math.exp(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # prepare p and v for the dot
        v_ptr = offsetv_y + tl.arange(0, BLOCK_N)[:, None]*HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
        v = tl.load(desc_v + v_ptr, mask=(tl.arange(0, BLOCK_N)+start_n)[:, None] < N_CTX, other=0.0)
        p = p.to(desc_v.dtype.element_ty)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N * HEAD_DIM
        offsetv_y += BLOCK_N * HEAD_DIM
        offseta_y += BLOCK_N
    return acc, l_i, m_i