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
        dtype=torch.float32
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
    ).to(tl.float32)
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
                    k_mat_ptr 
                    + offs_i[:, None] * k_stride_seq
                    + offs_d[None, :] * k_stride_dim
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
                mask=(offs_j[:, None] < limit_c) & (offs_d[None, :] < limit_d), 
                other=0.0
            ).to(tl.float32)

            # TODO: handle precision
            affinity += tl.dot(k_mat, kt_mat)

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
        ).to(tl.float32)
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
            res + offs_j * res_stride_kseq, 
            decay,
            mask=offs_j < limit_c
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
    src: torch.Tensor,
    dest: torch.Tensor,
    chunked_decay: torch.Tensor,
    chunk_size: int = 16,
    return_decay: bool = False,
    skip_preprocessing: bool = False,
):
    # TODO: check sizes
    b, nheads, qlen, qdim = q.shape
    _, kvheads, klen, _ = k.shape
    _, _, _, vdim = v.shape
    assert qlen == klen
    assert nheads % kvheads == 0
    assert qdim == vdim

    grid = (b, nheads, triton.cdiv(qlen, chunk_size))

    if not skip_preprocessing:
        # NOTE: move this somewhere?
        # L2-normalize K
        k = k / k.pow(2).sum(-1,True).sqrt().add(1e-6)

        # sigmoid
        src = src.sigmoid()
        dest = dest.sigmoid()

        # NOTE: 
        # - static_src and static_dest assumed to be sigmoided

    res = torch.zeros(
        (b, nheads, qlen, vdim), 
        device=q.device, 
        dtype=torch.float32
    ) 

    res_decay = None
    if return_decay:
        # NOTE: this should only be used for debugging
        res_decay = torch.zeros(
            (b, nheads, qlen, klen), 
            device=q.device, 
            dtype=torch.float32
        ) 

    _softmax_with_decay_fwd[grid](
        res, res_decay,
        q, k, v, src, dest,
        chunked_decay,
        res.stride(0), res.stride(1), res.stride(2), res.stride(3),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        chunked_decay.stride(0), chunked_decay.stride(1), chunked_decay.stride(2), chunked_decay.stride(3),
        src.stride(0), src.stride(1), src.stride(2),
        dest.stride(0), dest.stride(1), dest.stride(2),
        *(
            None if res_decay is None else
            res_decay.stride(i)
            for i in range(4)
        ),
        seqlen=klen,
        HEAD_DIM=qdim,
        chunk_size=chunk_size,
    )

    if return_decay:
        # NOTE: the upper tril entries not garanteed to be
        # correct due to causal nature of kernel
        return res, res_decay
    return res

@triton.autotune(
    [
        triton.Config({'BLOCK_C': 16, 'BLOCK_D': 16}, num_stages=1, num_warps=1),
    ],
    key=['BLOCK_C', 'BLOCK_D'],
)
@triton.jit
def _softmax_with_decay_fwd(
    res, res_decay,
    queries, keys, values, 
    src, dest,
    chunked_decay, 
    res_stride_b, res_stride_h, res_stride_seq, res_stride_dim,
    q_stride_b, q_stride_h, q_stride_seq, q_stride_dim,
    k_stride_b, k_stride_h, k_stride_seq, k_stride_dim,
    v_stride_b, v_stride_h, v_stride_seq, v_stride_dim,
    d_stride_b, d_stride_h, d_stride_chunk, d_stride_kseq,
    src_stride_b, src_stride_h, src_stride_seq, 
    dest_stride_b, dest_stride_h, dest_stride_seq, 
    res_decay_stride_b, res_decay_stride_h, res_decay_stride_qseq, 
    res_decay_stride_kseq,
    seqlen: int,
    chunk_size: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # dtype = desc_q.dtype.element_ty
    # assert BLOCK_C % chunk_size == 0

    pid_b = tl.program_id(0) # batch
    pid_h = tl.program_id(1) # query head
    pid_r = tl.program_id(2) # row chunk

    # offset by batch and head
    res += pid_b * res_stride_b + pid_h * res_stride_h
    queries += pid_b * q_stride_b + pid_h * q_stride_h
    keys += pid_b * k_stride_b + pid_h * k_stride_h
    values += pid_b * v_stride_b + pid_h * v_stride_h
    chunked_decay += pid_b * d_stride_b + pid_h * d_stride_h

    src += pid_b * src_stride_b + pid_h * src_stride_h
    dest += pid_b * dest_stride_b + pid_h * dest_stride_h

    # keys (row) and dest will be offset by chunk
    keys_r = keys + pid_r * chunk_size * k_stride_seq
    queries_r = queries + pid_r * chunk_size * q_stride_seq
    res_r = res + pid_r * chunk_size * res_stride_seq
    dest += pid_r * chunk_size * dest_stride_seq

    if res_decay:
        res_decay += pid_b * res_decay_stride_b + pid_h * res_decay_stride_h
        res_decay += pid_r * chunk_size * res_decay_stride_qseq

    # decay offset by chunk index
    decay_prev_chunk = chunked_decay # Need to set this otherwise I cannot
    if pid_r > 0:
        # get the previous row
        # decay_prev = decay + (pid_r-1) * d_stride_chunk
        decay_prev_chunk += (pid_r-1) * d_stride_chunk

    # online-softmax accumulation elements
    score_max = tl.zeros([chunk_size], dtype=tl.float32) - float("inf")
    score_denom = tl.zeros([chunk_size], dtype=tl.float32) + 1.0
    # TODO: its not gauranteed that value dim 
    # equals to query and key dim
    acc = tl.zeros([chunk_size, HEAD_DIM], dtype=tl.float32)

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
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = tl.arange(0, HEAD_DIM)
    limit_r = seqlen - pid_r * chunk_size # row limit
    limit_c = seqlen # col_limit

    # load the dest
    dest_vec = tl.load(
        dest + offs_i * dest_stride_seq,
        mask=offs_i < limit_r,
        other=0.0,
    ).to(tl.float32)
    dest_vec = tl.exp2(tl.log2(dest_vec) / 3.0) # pow (1/3)

    # process the columns
    for c in range(0, nC):

        affinity = tl.zeros((chunk_size, BLOCK_C), dtype=tl.float32)
        score = tl.zeros((chunk_size, BLOCK_C), dtype=tl.float32)

        # compute the keys (tiled on dimension)
        limit_d = HEAD_DIM # dims
        k_mat_ptr = keys_r
        q_mat_ptr = queries_r
        kt_mat_ptr = keys
        v_mat_ptr = values

        for _ in range(0, nD):

            # TODO: for the last block need to do the triangular masking
            # NOTE: this row load can actually be optimized
            # - since its repetitive, especially in the case nD == 1
            k_mat = tl.load(
                (
                    k_mat_ptr 
                    + offs_i[:, None] * k_stride_seq
                    + offs_d[None, :] * k_stride_dim
                ),
                mask=(offs_i[:, None] < limit_r) & (offs_d[None, :] < limit_d), 
                other=0.0
            ).to(tl.float32)
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
                mask=(offs_j[:, None] < limit_c) & (offs_d[None, :] < limit_d), 
                other=0.0
            ).to(tl.float32)

            # TODO: handle precision
            affinity += tl.dot(k_mat, kt_mat)
            score += tl.dot(q_mat, kt_mat)

            # handle the limit
            limit_d -= BLOCK_D

            # handle the pointers
            k_mat_ptr += BLOCK_D * k_stride_dim
            kt_mat_ptr += BLOCK_D * k_stride_dim
            q_mat_ptr += BLOCK_D * k_stride_dim

        # .relu().pow(2/3)
        affinity = tl.exp2(tl.log2(tl.maximum(affinity, 0.0)) * 2.0 / 3.0)

        # load the src
        src_vec = tl.load(
            src + offs_j * src_stride_seq,
            mask=offs_j < limit_c,
            other=0.0,
        ).to(tl.float32)
        src_vec = tl.exp2(tl.log2(src_vec) / 3.0) # pow (1/3)
        affinity = affinity * dest_vec[:, None] * src_vec[None, :]

        # - convert to log(1-p)
        # torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())
        decay = tl.log(1.0 - tl.clamp(affinity, 0.0, 1.0 - 1e-6)) 

        # SEE notes on this in the _chunked_delay kernel
        # if the rightmost element of the block
        # is greater
        offset = c * BLOCK_C - pid_r * chunk_size
        if offset >= 0 and chunk_size >= offset:
            decay = tl.where(
                (offs_i[:, None] > (offs_j[None, :] + offset)), 
                decay, 
                0.0 # dont set this to -inf yet because we need to cumsum
            )

        # cumsum over the chunk rows
        decay = tl.cumsum(decay, axis=-2)

        # in the causal case, the only possiblity to have
        # a block above with prev decay values, is if the 
        # coordinate ( pid_r * chunk_size, c * BLOCK_C )
        # satisfies
        # - c * BLOCK_C < pid_r * chunk_size
        # - this is equivalent to 
        #   offset > 0
        #   where offset = pid_r * chunk_size - c * BLOCK_C

        # NOTE: can combine this with the above offset

        offset = pid_r * chunk_size - c * BLOCK_C 
        # if decay_prev and offset > 0:
        if pid_r > 0 and offset > 0:
            decay_boundary = tl.load(
                (decay_prev_chunk + offs_j[None,:]),
                mask=(
                    offs_j[None, :] < offset
                ),
                other=0.0,
            ).to(tl.float32)

            # hopefully this can distribute across
            # rows
            decay += decay_boundary

            # increment pointer
            decay_prev_chunk += BLOCK_C * d_stride_kseq

        # NOTE: see above notes
        offset = c * BLOCK_C - pid_r * chunk_size
        if offset >= 0 and chunk_size >= offset:
            decay = tl.where(
                (offs_i[:, None] >= (offs_j[None, :] + offset)), 
                decay, 
                - float("inf"), # not set it to inf
            )

        if res_decay:
            # NOTE: also consider returning score + delay
            # if the user wants to inspect the values
            tl.store(
                (
                    res_decay 
                    + offs_i[:, None] * res_decay_stride_qseq
                    + offs_j[None,:] * res_decay_stride_kseq
                ),
                decay,
                mask=(
                    (offs_i[:, None] < limit_r)
                    & (offs_j[None, :] < limit_c)
                )
            )
            res_decay += BLOCK_C * res_decay_stride_kseq

        # ---------- ONLINE SOFTMAX (FLASH ATTENTION) -------------
        score += decay

        # Stabilize logsumexp using the subtract max trick
        score_max_prev = score_max # m_{i-1}
        score_denom_prev = score_denom # d_{i-1}
        score_max = tl.maximum(
            score_max, 
            tl.max(score, axis=1) , 
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

        # - update score denom
        # d_i = d_{i-1} * exp(m_{i-1} - m_i) + \sum_{j} exp(q^T k - m_i)
        score_denom = (
            score_denom_corrected + tl.sum(weights, axis=1)
        )

        v_mat = tl.load(
            (
                v_mat_ptr 
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

        # move pointer with column chunk
        keys += BLOCK_C * k_stride_seq
        values += BLOCK_C * v_stride_seq
        src += BLOCK_C * src_stride_seq

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
