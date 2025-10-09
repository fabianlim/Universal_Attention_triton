import torch
import triton
import triton.language as tl

CHUNK_SIZE = 16 # TODO: tune this

def chunked_decay(
    keys: torch.Tensor, # b,h,l,d
    src: torch.Tensor,
    dest: torch.Tensor,
    skip_preprocessing: bool = False,
    chunk_size: int = None,
):

    if chunk_size is None:
        chunk_size = CHUNK_SIZE


    b, kvheads, klen, kdim = keys.shape
    num_chunks = triton.cdiv(klen, chunk_size)
    res = torch.zeros(
        (b, kvheads, num_chunks, klen), 
        device=keys.device, 
        dtype=torch.float32
    )

    grid = (b, kvheads, num_chunks)

    if not skip_preprocessing:

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
        # - so in the case j=0, check i > offset
        # - so we check chunk_size - 1 > offset
        # - same as chunk_size >= offset

        # if the rightmost element of the block
        # is greater
        offset = c * BLOCK_C - pid_r * chunk_size
        if chunk_size >= offset:
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
    chunk_size: int = None,
    return_denom: bool = False,
    return_decay: bool = False,
    skip_preprocessing: bool = False,
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
    res_denom = torch.zeros(
        (b, nheads, qlen), 
        device=q.device, 
        dtype=torch.float32
    ) 

    res_decay = None
    if return_decay:
        # NOTE: this should only be used for debugging
        res_decay = torch.zeros(
            (b, kvheads, qlen, klen), 
            device=q.device, 
            dtype=torch.float32
        ) 

    _softmax_with_decay_fwd[grid](
        res, 
        res_denom, 
        res_decay,
        q, k, v, src, dest,
        chunked_decay,
        res.stride(0), res.stride(1), res.stride(2), res.stride(3),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        chunked_decay.stride(0), chunked_decay.stride(1), chunked_decay.stride(2), chunked_decay.stride(3),
        src.stride(0), src.stride(1), src.stride(2),
        dest.stride(0), dest.stride(1), dest.stride(2),
        res_denom.stride(0), res_denom.stride(1), res_denom.stride(2),
        *(
            None if res_decay is None else
            res_decay.stride(i)
            for i in range(4)
        ),
        seqlen=klen,
        HEAD_DIM=qdim,
        group_size=nheads // kvheads,
        chunk_size=chunk_size,
    )

    if return_denom:
        # this takes precedence over decay
        return res, res_denom

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
    res, 
    res_denom, 
    res_decay,
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
    res_denom_stride_b, res_denom_stride_h, res_denom_stride_seq, 
    res_decay_stride_b, res_decay_stride_h, res_decay_stride_qseq, 
    res_decay_stride_kseq,
    seqlen: int,
    group_size: int,
    chunk_size: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # dtype = desc_q.dtype.element_ty
    # assert BLOCK_C % chunk_size == 0

    pid_b = tl.program_id(0) # batch
    pid_h = tl.program_id(1) # query head
    hkv = pid_h // group_size # key / value head
    pid_r = tl.program_id(2) # row chunk

    # offset by batch and head
    res += pid_b * res_stride_b + pid_h * res_stride_h
    res_denom += pid_b * res_denom_stride_b + pid_h * res_denom_stride_h
    queries += pid_b * q_stride_b + pid_h * q_stride_h
    keys += pid_b * k_stride_b + hkv * k_stride_h
    values += pid_b * v_stride_b + hkv * v_stride_h
    chunked_decay += pid_b * d_stride_b + hkv * d_stride_h

    src += pid_b * src_stride_b + hkv * src_stride_h
    dest += pid_b * dest_stride_b + hkv * dest_stride_h

    # keys (row) and dest will be offset by chunk
    keys_r = keys + pid_r * chunk_size * k_stride_seq
    queries_r = queries + pid_r * chunk_size * q_stride_seq
    res_r = res + pid_r * chunk_size * res_stride_seq
    res_denom_r = res_denom + pid_r * chunk_size * res_denom_stride_seq
    dest += pid_r * chunk_size * dest_stride_seq

    if res_decay:
        res_decay += pid_b * res_decay_stride_b + hkv * res_decay_stride_h
        res_decay += pid_r * chunk_size * res_decay_stride_qseq

    # decay offset by chunk index
    decay_prev_chunk = chunked_decay # Need to set this otherwise I cannot
    if pid_r > 0:
        # get the previous row
        # decay_prev = decay + (pid_r-1) * d_stride_chunk
        decay_prev_chunk += (pid_r-1) * d_stride_chunk

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
                mask= (offs_d[:, None] < limit_d) & (offs_j[None, :] < limit_c),
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
        if chunk_size >= offset:
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
        if chunk_size >= offset:
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

    # this is the equiv norm without the 
    # max normalization
    tl.store(
        res_denom_r + offs_i * res_denom_stride_seq,
        score_max + tl.log(score_denom),
        mask=offs_i < limit_r
    )

def rowwise_bwd(
    dout: torch.Tensor, # b,h,l,d
    q: torch.Tensor, # b,h,l,d
    k: torch.Tensor, # b,h,l,d
    v: torch.Tensor, # b,h,l,d
    src: torch.Tensor,
    dest: torch.Tensor,
    chunked_decay: torch.Tensor,
    score_denom: torch.Tensor,
    chunk_size: int = None,
    skip_preprocessing: bool = False,
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

    res_dQ = torch.zeros(
        (b, nheads, qlen, qdim), 
        device=q.device, 
        dtype=torch.float32
    ) 

    res_dK1 = torch.zeros(
        (b, kvheads, klen, qdim), 
        device=q.device, 
        dtype=torch.float32
    ) 

    res_dZscore = torch.zeros(
        (b, nheads, qlen), 
        device=q.device, 
        dtype=torch.float32
    ) 

    num_chunks = triton.cdiv(qlen, chunk_size)
    res_dY_chunked = torch.zeros(
        (b, nheads, num_chunks, qlen), 
        device=q.device, 
        dtype=torch.float32
    ) 

    res_ddest = torch.zeros(
        (b, kvheads, qlen), 
        device=q.device, 
        dtype=torch.float32
    ) 

    # NOTE: need to run this thrice, dont have a good way
    # to get both res_dZscore and res_dY_chunked in a single run
    for p in [1,2,3]:

        if p == 3:
            # for the third pass, we need to accumulate
            # this (so it becomes dZ1_chunked)
            res_dY_chunked = res_dY_chunked.flip(-2).cumsum(dim=-2).flip(-2)

        _rowwise_bwd[grid](
            res_dQ, 
            res_dK1,
            res_dZscore,
            res_dY_chunked,
            res_ddest,
            dout, q, k, v, src, dest,
            chunked_decay,
            score_denom,
            res_dQ.stride(0), res_dQ.stride(1), res_dQ.stride(2), res_dQ.stride(3),
            res_dK1.stride(0), res_dK1.stride(1), res_dK1.stride(2), res_dK1.stride(3),
            res_dZscore.stride(0), res_dZscore.stride(1), res_dZscore.stride(2), 
            res_dY_chunked.stride(0), res_dY_chunked.stride(1), res_dY_chunked.stride(2), res_dY_chunked.stride(3),
            res_ddest.stride(0), res_ddest.stride(1), res_ddest.stride(2),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            chunked_decay.stride(0), chunked_decay.stride(1), chunked_decay.stride(2), chunked_decay.stride(3),
            score_denom.stride(0), score_denom.stride(1), score_denom.stride(2),
            src.stride(0), src.stride(1), src.stride(2),
            dest.stride(0), dest.stride(1), dest.stride(2),
            seqlen=klen,
            HEAD_DIM=qdim,
            group_size=nheads // kvheads,
            chunk_size=chunk_size,
            PASS=p,
        )

    return res_dQ, res_dK1, res_dZscore, res_dY_chunked, res_ddest

@triton.autotune(
    [
        triton.Config({'BLOCK_C': 16, 'BLOCK_D': 16}, num_stages=1, num_warps=1),
    ],
    key=['BLOCK_C', 'BLOCK_D'],
)
@triton.jit
def _rowwise_bwd(
    res_dQ, 
    res_dK1, 
    res_dZsc,
    res_dYc,
    res_ddest,
    dout, 
    queries, keys, values, 
    src, dest,
    chunked_decay, 
    score_denom,
    res_dQ_stride_b, res_dQ_stride_h, res_dQ_stride_qseq, 
    res_dQ_stride_dim,
    res_dK1_stride_b, res_dK1_stride_h, res_dK1_stride_seq, 
    res_dK1_stride_dim,
    res_dZsc_stride_b, res_dZsc_stride_h, res_dZsc_stride_seq, 
    res_dYc_stride_b, res_dYc_stride_h, res_dYc_stride_chunk, res_dYc_stride_seq, 
    res_dd_stride_b, res_dd_stride_h, res_dd_stride_seq, 
    do_stride_b, do_stride_h, do_stride_seq, do_stride_dim,
    q_stride_b, q_stride_h, q_stride_seq, q_stride_dim,
    k_stride_b, k_stride_h, k_stride_seq, k_stride_dim,
    v_stride_b, v_stride_h, v_stride_seq, v_stride_dim,
    d_stride_b, d_stride_h, d_stride_chunk, d_stride_kseq,
    denom_stride_b, denom_stride_h, denom_stride_seq, 
    src_stride_b, src_stride_h, src_stride_seq, 
    dest_stride_b, dest_stride_h, dest_stride_seq, 
    seqlen: int,
    group_size: int,
    chunk_size: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PASS: tl.constexpr,
):

    pid_b = tl.program_id(0) # batch
    pid_h = tl.program_id(1) # query head
    hkv = pid_h // group_size # key / value head
    pid_r = tl.program_id(2) # row chunk

    # offset by batch and head
    res_dQ += pid_b * res_dQ_stride_b + pid_h * res_dQ_stride_h
    res_dK1 += pid_b * res_dK1_stride_b + hkv * res_dK1_stride_h
    res_dZsc += pid_b * res_dZsc_stride_b + pid_h * res_dZsc_stride_h
    res_dYc += pid_b * res_dYc_stride_b + pid_h * res_dYc_stride_h
    res_ddest += pid_b * res_dd_stride_b + hkv * res_dd_stride_h
    dout += pid_b * do_stride_b + pid_h * do_stride_h
    queries += pid_b * q_stride_b + pid_h * q_stride_h
    keys += pid_b * k_stride_b + hkv * k_stride_h
    values += pid_b * v_stride_b + hkv * v_stride_h
    chunked_decay += pid_b * d_stride_b + hkv * d_stride_h
    score_denom += pid_b * denom_stride_b + hkv * denom_stride_h

    src += pid_b * src_stride_b + hkv * src_stride_h
    dest += pid_b * dest_stride_b + hkv * dest_stride_h

    # keys (row) and dest will be offset by chunk
    keys_r = keys + pid_r * chunk_size * k_stride_seq
    queries_r = queries + pid_r * chunk_size * q_stride_seq
    dout_r = dout + pid_r * chunk_size * do_stride_seq
    denom_r = score_denom + pid_r * chunk_size * denom_stride_seq
    res_dQ_r = res_dQ + pid_r * chunk_size * res_dQ_stride_qseq
    res_dK1_r = res_dK1 + pid_r * chunk_size * res_dK1_stride_seq
    res_dZsc_r = res_dZsc + pid_r * chunk_size * res_dZsc_stride_seq
    res_dYc_r = res_dYc + pid_r * res_dYc_stride_chunk
    res_dd_r = res_ddest + pid_r * chunk_size * res_dd_stride_seq
    dest += pid_r * chunk_size * dest_stride_seq

    # decay offset by chunk index
    decay_prev_chunk = chunked_decay # Need to set this otherwise I cannot
    if pid_r > 0:
        # get the previous row
        decay_prev_chunk += (pid_r-1) * d_stride_chunk

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

    # we need two passes 
    if PASS == 1:
        # in the first pass we compute dZScore
        dZScore_sum = tl.zeros([chunk_size], dtype=tl.float32)

        # TODO: its not gauranteed that value dim 
        # equals to query and key dim
        acc = None
        acc3 = None
        acc4 = None

    elif PASS == 2:
        # in the 2nd pass we load dZScore
        # to compute dY_chunked
        dZScore_sum = tl.load(
            res_dZsc_r + offs_i * res_dZsc_stride_seq,
            mask=offs_i < limit_r,
            other=0.0,
        ).to(tl.float32)

        # for dQ
        acc = tl.zeros([chunk_size, HEAD_DIM], dtype=tl.float32)

        # not needed
        acc3 = None
        acc4 = None

    elif PASS == 3:

        # in the 3rd pass, we load dZScore
        # to compute dY
        dZScore_sum = tl.load(
            res_dZsc_r + offs_i * res_dZsc_stride_seq,
            mask=offs_i < limit_r,
            other=0.0,
        ).to(tl.float32)

        # not needed
        acc = None

        # for ddest
        acc3 = tl.zeros([chunk_size], dtype=tl.float32)
        acc4 = tl.zeros([chunk_size, HEAD_DIM], dtype=tl.float32)


    # load the dest
    dest_vec = tl.load(
        dest + offs_i * dest_stride_seq,
        mask=offs_i < limit_r,
        other=0.0,
    ).to(tl.float32)
    dest_vec = tl.exp2(tl.log2(dest_vec) / 3.0) # pow (1/3)

    # load the denom
    denom_vec = tl.load(
        denom_r + offs_i * res_dZsc_stride_seq,
        mask=offs_i < limit_r,
        other=0.0,
    ).to(tl.float32)

    # process the columns
    for c in range(0, nC):

        affinity = tl.zeros((chunk_size, BLOCK_C), dtype=tl.float32)
        score = tl.zeros((chunk_size, BLOCK_C), dtype=tl.float32)
        dZ = tl.zeros((chunk_size, BLOCK_C), dtype=tl.float32)

        # -------dY,dQ, dZ,dZScore -------

        # compute the keys (tiled on dimension)
        limit_d = HEAD_DIM # dims
        k_mat_ptr = keys_r 
        q_mat_ptr = queries_r 
        do_mat_ptr = dout_r
        kt_mat_ptr = keys
        vt_mat_ptr = values
        kc_mat_ptr = keys # for dQ

        for _ in range(0, nD):

            # TODO: for the last block need to do the triangular masking
            # NOTE: this row load can actually be optimized
            # - since its repetitive, especially in the case nD == 1
            # - for decay
            k_mat = tl.load(
                (
                    k_mat_ptr 
                    + offs_i[:, None] * k_stride_seq
                    + offs_d[None, :] * k_stride_dim
                ),
                mask=(offs_i[:, None] < limit_r) & (offs_d[None, :] < limit_d), 
                other=0.0
            ).to(tl.float32)

            # for score -> dY
            q_mat = tl.load(
                (
                    q_mat_ptr 
                    + offs_i[:, None] * q_stride_seq
                    + offs_d[None, :] * q_stride_dim
                ),
                mask=(offs_i[:, None] < limit_r) & (offs_d[None, :] < limit_d), 
                other=0.0
            ).to(tl.float32)

            # for decay
            kt_mat = tl.load(
                (
                    kt_mat_ptr 
                    + offs_d[:, None] * k_stride_dim
                    + offs_j[None, :] * k_stride_seq
                ),
                mask= (offs_d[:, None] < limit_d) & (offs_j[None, :] < limit_c),
                other=0.0
            ).to(tl.float32)

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

            # TODO: handle precision
            affinity += tl.dot(k_mat, kt_mat)
            score += tl.dot(q_mat, kt_mat)
            dZ += tl.dot(do_mat, vt_mat)

            # handle the limit
            limit_d -= BLOCK_D

            # handle the pointers
            k_mat_ptr += BLOCK_D * k_stride_dim
            kt_mat_ptr += BLOCK_D * k_stride_dim
            q_mat_ptr += BLOCK_D * q_stride_dim
            do_mat_ptr += BLOCK_D * do_stride_dim
            vt_mat_ptr += BLOCK_D * v_stride_dim

        affinity0 = affinity # store
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
        if chunk_size >= offset:
            decay = tl.where(
                (offs_i[:, None] > (offs_j[None, :] + offset)), 
                decay, 
                0.0 # dont set this to -inf yet because we need to cumsum
            )

        # cumsum over the chunk rows
        decay = tl.cumsum(decay, axis=-2)

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
        if chunk_size >= offset:
            decay = tl.where(
                (offs_i[:, None] >= (offs_j[None, :] + offset)), 
                decay, 
                - float("inf"), # not set it to inf
            )

        # ---------- COMPUTE dZScore -------------
        score += decay
        score -= denom_vec[:, None] # take into account the denominator

        # - need to do this for the last chunk ends
        # - so that the below score will not 
        #   take those values into account
        #  x x x |  0    0
        #  x x x |  0    0 
        #  0 0 0 |  0    0 

        # NO NEED FOR this since dZ = dout @ vT will have zeros
        # in the right places (thanks to tl.load)
        # - but just handle it in the score

        if limit_c < BLOCK_C:
            score = tl.where(
                offs_j[None, :] < limit_c,
                score, 0.
            )

        if limit_r < chunk_size:
            score = tl.where(
                offs_i[:, None] < limit_r,
                score, 0.0
            )

        # dZScore
        dZScore = dZ * tl.exp(score)

        if PASS == 1:
            # - in pass 1, we compute 
            # dZScoreSum
            dZScore_sum += tl.sum(dZScore, axis=1)
        elif PASS == 2:
            
            # in pass 2 need to compute 
            # _dzScore = (dZ * score).sum(-1, keepdim=True) # see notes above

            k_mat = tl.load(
                (
                    kc_mat_ptr 
                    + offs_j[:, None] * k_stride_seq
                    + offs_v[None, :] * k_stride_dim # NOTE: assumed same
                ),
                mask=(offs_j[:, None] < limit_c),
                other=0.0
            ).to(tl.float32)

            # first term (score * dZ)
            # - dZ should have the appropriate zeros in the boundary
            acc += tl.dot(
                dZScore - tl.exp(score) * dZScore_sum[:, None], 
                k_mat
            )

            # in pass 2, we compute the chunked
            # dY, 
            # NOTE: we need two passes as there
            # is no good way to compute this sum
            # while simultanously computing dZScore_sum
            
            # store the quantities for dY
            tl.store(
                res_dYc_r + offs_j * res_dYc_stride_seq, 
                tl.sum(
                    (
                        dZScore 
                        - tl.exp(score) * dZScore_sum[:, None]
                    ),
                    axis=-2
                ),
                mask=offs_j < limit_c
            )
        elif PASS == 3:
            # in the 3rd pass we load dY_chunked
            # which we assume it has been accumulated
            # dZ1_chunked = dY_chunked.flip(-2).cumsum(dim=-2).flip(-2)

            dZ1_boundary = tl.load(
                res_dYc_r + offs_j,
                mask=(
                    offs_j < limit_c
                ),
                other=0.0,
            ).to(tl.float32)

            # dY (new)
            dY = dZScore - tl.exp(score) * dZScore_sum[:, None]

            # dZ1
            # - NOTE: i need to rotate dY.. dont have a good 
            # way to do it
            dYrotate = tl.dot(
                tl.where(
                    (offs_i[:, None] - offs_i[None, :]) == 1,
                    1.0, 0.0
                ),
                dY
            )
            dZ1 = tl.where(
                (offs_i[:, None] > (offs_j[None, :] + offset)), 
                dZ1_boundary[None, :] - tl.cumsum(dYrotate, axis=-2),
                0.0
            )

            # Z2 = deltanet_relu2 * ds 
            # term = Z2.pow(2/3) - Z2
            #      = x^2 - x^3
            #      = x^2 (1 - x)
            # where x = affinity, and 0 <= x <= 1

            # so then ddest is 
            #   dZ2 * src * deletanet_relu2
            #       = x^3 / dest
            # where 
            #   dZ2 = -dZ1 / (3 * term)
            # so:
            #   ddest = - dZ1 * x^3 / (3 * dest * term)
            #         = - dZ1 * x^3 / (3 * x^2 (1-x) * dest)
            #         = - dZ1 * x / (3 * (1-x) * dest)

            # - for ddest
            A = -dZ1 / 3 * tl.exp(
                tl.log(tl.clamp(affinity, 1e-4, 1.0))

                # this is not decay
                - tl.log(1.0 - tl.clamp(affinity, 0.0, 1.0 - 1e-6)) 

                # recall it had pow(1/3) applied
                - 3 * tl.log(tl.clamp(dest_vec[:, None], 1e-4, 1.0)) 
            ) 

            # ddest needs to be reduced over cols
            acc3 += tl.sum(A, axis=-1)

            # for dK1
            # dZ3 = torch.where(
            #     deltanet >= 0,
            #     2 * dZ2 * ds * deltanet,
            #     0.
            # ) # b,h,l,l

            # - recall above affinity0 is deltanet
            # - affinity1 is deltanet_relu2
            # dZ2 * ds 
            #  = - dZ1 * x^3 / (3 * term * deltanet_relu2)
            #
            # where x^3 = ds * deltanet_relu2
            # 
            # Thus, 2 * dZ3 
            #  = 2 * dZ2 * ds * deltanet
            #  = - dZ1 * x^3 * deltanet / (3 * term * deltanet_relu2)
            #
            # where we compute 2 * dZ3 only for deltanet > 0
            # so -> deltanet / deltanet_relu2
            #    => delenet / deltanet^2 (when deltanet > 0)
            #    => 1 / deltanet
            # NOTE: this computation is not very stable so 
            # the variance tends to be high
            B = -dZ1 * 2 / 3 * tl.exp(
                tl.log(tl.clamp(affinity, 1e-4, 1.0))

                # this is not decay
                - tl.log(1.0 - tl.clamp(affinity, 0.0, 1.0 - 1e-6)) 

                # see the explaination above
                - tl.log(
                    tl.maximum(affinity0, 1e-3)
                ) 
            ) 

            # this is 2 * dZ3
            B = tl.where(affinity0 > 0, B, 0.)

            k_mat = tl.load(
                (
                    keys 
                    + offs_j[:, None] * k_stride_seq
                    + offs_v[None, :] * k_stride_dim
                ),
                mask=(offs_j[:, None] < limit_c),
                other=0.0
            ).to(tl.float32)

            # dK1
            acc4 += tl.dot(B, k_mat)

            # if pid_r == 0 and pid_h == 0 and c == 0:
            #     # print ("dZ1", dZ1)
            #     print ("acc3", acc3)

        # move pointer with column chunk
        keys += BLOCK_C * k_stride_seq
        values += BLOCK_C * v_stride_seq
        src += BLOCK_C * src_stride_seq
        res_dYc_r += BLOCK_C * res_dYc_stride_seq

        # handle the limit
        limit_c -= BLOCK_C

    # -  DONE WITH COL CHUNK LOOPS - 

    if PASS == 1:
        # we compute dZScore_sum
        tl.store(
            res_dZsc_r + offs_i * res_dZsc_stride_seq,
            dZScore_sum,
            mask=offs_i < limit_r
        )
    elif PASS == 2:
        # in PASS 2 we compute dQ 
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


    elif PASS == 3:
        # in pass 3 we compute
        # ddest
        tl.store(
            res_dd_r + offs_i * res_dd_stride_seq,
            acc3,
            mask=offs_i < limit_r
        )

        # dK1
        tl.store(
            (
                res_dK1_r 
                + offs_i[:, None] * res_dK1_stride_seq
                + offs_v[None, :] * res_dK1_stride_dim
            ),
            acc4,
            mask=(
                (offs_i[:, None] < limit_r)
            )
        )

def colwise_bwd(
    dout: torch.Tensor, # b,h,l,d
    q: torch.Tensor, # b,h,l,d
    k: torch.Tensor, # b,h,l,d
    v: torch.Tensor, # b,h,l,d
    src: torch.Tensor,
    dest: torch.Tensor,
    score_denom: torch.Tensor,
    dZScoreSum: torch.Tensor,
    chunked_dZ1: torch.Tensor,
    chunk_size: int = None,
    skip_preprocessing: bool = False,
):

    if chunk_size is None:
        chunk_size = CHUNK_SIZE

    # TODO: check sizes
    b, nheads, qlen, qdim = q.shape
    _, kvheads, klen, _ = k.shape
    _, _, vlen, vdim = v.shape
    assert qlen == klen
    assert qlen == vlen
    assert nheads % kvheads == 0
    assert qdim == vdim

    grid = (b, nheads, triton.cdiv(klen, chunk_size))

    if not skip_preprocessing:
        # NOTE: move this somewhere?
        # L2-normalize K
        k = k / k.pow(2).sum(-1,True).sqrt().add(1e-6)

        # sigmoid
        src = src.sigmoid()
        dest = dest.sigmoid()

        # NOTE: 
        # - static_src and static_dest assumed to be sigmoided

    res_dK2 = torch.zeros(
        (b, kvheads, qlen, qdim), 
        device=q.device, 
        dtype=torch.float32
    ) 

    res_dV = torch.zeros(
        (b, kvheads, vlen, vdim), 
        device=q.device, 
        dtype=torch.float32
    ) 

    res_dsrc = torch.zeros(
        (b, kvheads, qlen), 
        device=q.device, 
        dtype=torch.float32
    ) 

    _colwise_bwd[grid](
        res_dK2,
        res_dV,
        res_dsrc,
        dout, q, k, v, src, dest,
        chunked_dZ1,
        score_denom,
        dZScoreSum,
        res_dK2.stride(0), res_dK2.stride(1), res_dK2.stride(2), res_dK2.stride(3),
        res_dV.stride(0), res_dV.stride(1), res_dV.stride(2), res_dV.stride(3),
        res_dsrc.stride(0), res_dsrc.stride(1), res_dsrc.stride(2),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dZScoreSum.stride(0), dZScoreSum.stride(1), dZScoreSum.stride(2),
        score_denom.stride(0), score_denom.stride(1), score_denom.stride(2),
        src.stride(0), src.stride(1), src.stride(2),
        dest.stride(0), dest.stride(1), dest.stride(2),
        chunked_dZ1.stride(0), chunked_dZ1.stride(1), chunked_dZ1.stride(2), chunked_dZ1.stride(3),
        seqlen=klen,
        HEAD_DIM=qdim,
        group_size=nheads // kvheads,
        chunk_size=chunk_size,
    )

    return res_dK2, res_dV, res_dsrc

@triton.autotune(
    [
        triton.Config({'BLOCK_R': 16, 'BLOCK_D': 16}, num_stages=1, num_warps=1),
    ],
    key=['BLOCK_R', 'BLOCK_D'],
)
@triton.jit
def _colwise_bwd(
    res_dK2, 
    res_dV,
    res_dsrc,
    dout, 
    queries, keys, values, 
    src, dest,
    chunked_dZ1,
    score_denom,
    dZScoreSum, 
    res_dK2_stride_b, res_dK2_stride_h, res_dK2_stride_seq, 
    res_dK2_stride_dim,
    res_dV_stride_b, res_dV_stride_h, res_dV_stride_seq, 
    res_dV_stride_dim,
    res_dsrc_stride_b, res_dsrc_stride_h, res_dsrc_stride_seq, 
    do_stride_b, do_stride_h, do_stride_seq, do_stride_dim,
    q_stride_b, q_stride_h, q_stride_seq, q_stride_dim,
    k_stride_b, k_stride_h, k_stride_seq, k_stride_dim,
    v_stride_b, v_stride_h, v_stride_seq, v_stride_dim,
    dZS_stride_b, dZS_stride_h, dZS_stride_seq,
    denom_stride_b, denom_stride_h, denom_stride_seq, 
    src_stride_b, src_stride_h, src_stride_seq, 
    dest_stride_b, dest_stride_h, dest_stride_seq, 
    dz1_stride_b, dz1_stride_h, dz1_stride_chunk, dz1_stride_seq,
    seqlen: int,
    group_size: int,
    chunk_size: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):

    pid_b = tl.program_id(0) # batch
    pid_h = tl.program_id(1) # query head
    hkv = pid_h // group_size # key / value head
    pid_c = tl.program_id(2) # col chunk

    # offset by batch and head
    res_dV += pid_b * res_dV_stride_b + hkv * res_dV_stride_h
    res_dK2 += pid_b * res_dK2_stride_b + hkv * res_dK2_stride_h
    res_dsrc += pid_b * res_dsrc_stride_b + hkv * res_dsrc_stride_h
    dout += pid_b * do_stride_b + pid_h * do_stride_h
    queries += pid_b * q_stride_b + pid_h * q_stride_h
    dZScoreSum += pid_b * dZS_stride_b + pid_h * dZS_stride_h
    keys += pid_b * k_stride_b + hkv * k_stride_h
    values += pid_b * v_stride_b + hkv * v_stride_h
    score_denom += pid_b * denom_stride_b + hkv * denom_stride_h

    # - this is the cumulative chunked dY
    chunked_dZ1 += pid_b * dz1_stride_b + pid_h * dz1_stride_h

    src += pid_b * src_stride_b + hkv * src_stride_h
    dest += pid_b * dest_stride_b + hkv * dest_stride_h

    # - positition on the main block diagonal

    # keys (col) and dest will be offset by chunk
    keys_c = keys + pid_c * chunk_size * k_stride_seq
    queries_r = queries + pid_c * chunk_size * q_stride_seq
    values_c = values + pid_c * chunk_size * v_stride_seq
    dout_r = dout + pid_c * chunk_size * do_stride_seq
    denom_r = score_denom + pid_c * chunk_size * denom_stride_seq
    dZS_r = dZScoreSum + pid_c * chunk_size * dZS_stride_seq
    res_dK2_c = res_dK2 + pid_c * chunk_size * res_dK2_stride_seq
    res_dV_c = res_dV + pid_c * chunk_size * res_dV_stride_seq
    res_src_c = res_dsrc + pid_c * chunk_size * res_dsrc_stride_seq

    # - since we start on the block diag
    #   we begin from here
    src += pid_c * chunk_size * dest_stride_seq
    dest += pid_c * chunk_size * dest_stride_seq

    # assume that block size devices chunk size
    tl.static_assert(chunk_size % BLOCK_R == 0)
    cfac = chunk_size / BLOCK_R

    # decay offset by chunk index
    decay_prev_chunk = tl.zeros([chunk_size], dtype=tl.float32)

    # we allow row chunk size to differ from column chunk,
    # so we 
    # - a the pid_r-th row chunk will take up (pid_r * chunk_size) columns
    # - so 
    nR = tl.cdiv(
        seqlen - pid_c * chunk_size, BLOCK_R
    ) # number of column chunks to process (including the spillover)
    nD = tl.cdiv(HEAD_DIM, BLOCK_D)

    # load scales
    offs_i = tl.arange(0, BLOCK_R) # rows
    offs_j = tl.arange(0, chunk_size)
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = tl.arange(0, HEAD_DIM)
    limit_r = seqlen # row_limit
    limit_c = seqlen - pid_c * chunk_size # col limit

    # load the dZ1_boundary
    chunked_dZ1 += pid_c * dz1_stride_chunk 
    chunked_dZ1 += pid_c * chunk_size * dz1_stride_seq
    dZ1_boundary = tl.load(
        (chunked_dZ1 + offs_j),
        mask=(
            offs_j < limit_c
        ),
        other=0.0,
    ).to(tl.float32)

    # computation of dK_2
    acc = tl.zeros([HEAD_DIM, chunk_size], dtype=tl.float32)

    # computation of dV
    acc2 = tl.zeros([HEAD_DIM, chunk_size], dtype=tl.float32)

    # computation of dsrc
    acc3 = tl.zeros([chunk_size], dtype=tl.float32)

    # load the src
    src_vec = tl.load(
        src + offs_j * src_stride_seq,
        mask=offs_j < limit_c,
        other=0.0,
    ).to(tl.float32)
    src_vec = tl.exp2(tl.log2(src_vec) / 3.0) # pow (1/3)

    # duplicate
    keys_r = keys_c

    # process the rows
    for r in range(0, nR):

        affinity = tl.zeros((BLOCK_R, chunk_size), dtype=tl.float32)
        score = tl.zeros((BLOCK_R, chunk_size), dtype=tl.float32)
        dZ = tl.zeros((BLOCK_R, chunk_size), dtype=tl.float32)

        # ------- dZ -> dY -------

        # compute the keys (tiled on dimension)
        limit_d = HEAD_DIM # dims

        k_mat_ptr = keys_r
        q_mat_ptr = queries_r 
        kt_mat_ptr = keys_c
        do_mat_ptr = dout_r
        vt_mat_ptr = values_c
        qc_mat_ptr = queries_r # for dK_2

        for _ in range(0, nD):

            k_mat = tl.load(
                (
                    k_mat_ptr 
                    + offs_i[:, None] * k_stride_seq
                    + offs_d[None, :] * k_stride_dim
                ),
                mask=(offs_i[:, None] < limit_r) & (offs_d[None, :] < limit_d), 
                other=0.0
            ).to(tl.float32)

            # for score -> dY
            q_mat = tl.load(
                (
                    q_mat_ptr 
                    + offs_i[:, None] * q_stride_seq
                    + offs_d[None, :] * q_stride_dim
                ),
                mask=(offs_i[:, None] < limit_r) & (offs_d[None, :] < limit_d), 
                other=0.0
            ).to(tl.float32)

            # for decay
            kt_mat = tl.load(
                (
                    kt_mat_ptr 
                    + offs_d[:, None] * k_stride_dim
                    + offs_j[None, :] * k_stride_seq
                ),
                mask= (offs_d[:, None] < limit_d) & (offs_j[None, :] < limit_c),
                other=0.0
            ).to(tl.float32)

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

            # TODO: handle precision
            affinity += tl.dot(k_mat, kt_mat)
            score += tl.dot(q_mat, kt_mat)
            dZ += tl.dot(do_mat, vt_mat)

            # handle the limit
            limit_d -= BLOCK_D

            # handle the pointers
            k_mat_ptr += BLOCK_D * k_stride_dim
            kt_mat_ptr += BLOCK_D * k_stride_dim
            q_mat_ptr += BLOCK_D * q_stride_dim
            do_mat_ptr += BLOCK_D * do_stride_dim
            vt_mat_ptr += BLOCK_D * v_stride_dim

        # .relu().pow(2/3)
        affinity = tl.exp2(tl.log2(tl.maximum(affinity, 0.0)) * 2.0 / 3.0)

        # load the dest
        dest_vec = tl.load(
            dest + offs_i * dest_stride_seq,
            mask=offs_i < limit_r,
            other=0.0,
        ).to(tl.float32)
        dest_vec = tl.exp2(tl.log2(dest_vec) / 3.0) # pow (1/3)

        # load the denom
        denom_vec = tl.load(
            denom_r + offs_i * denom_stride_seq,
            mask=offs_i < limit_r,
            other=0.0,
        ).to(tl.float32)

        affinity = affinity * dest_vec[:, None] * src_vec[None, :]

        # - convert to log(1-p)
        # torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())
        decay = tl.log(1.0 - tl.clamp(affinity, 0.0, 1.0 - 1e-6)) 

        # SEE notes on this in the _chunked_delay kernel
        # if the rightmost element of the block
        # is greater
        # - since r is offset by pid_c, we need to add
        # offset = pid_c * chunk_size - r * BLOCK_R - pid_c * chunk_size
        offset = - r * BLOCK_R 
        decay = tl.where(
            (offs_i[:, None] > (offs_j[None, :] + offset)), 
            decay, 
            0.0 # dont set this to -inf yet because we need to cumsum
        )

        # cumsum over the chunk rows
        chunk_decay_sum = tl.sum(decay, axis=-2)
        decay = tl.cumsum(decay, axis=-2) + decay_prev_chunk
        # accumulate decay (for next upcoming chunk)
        decay_prev_chunk += chunk_decay_sum

        # same offset as above
        decay = tl.where(
            (offs_i[:, None] >= (offs_j[None, :] + offset)), 
            decay, 
            - float("inf"), # not set it to inf
        )

        # ---------- COMPUTE dZScore -------------
        score += decay
        score -= denom_vec[:, None] # take into account the denominator

        # - need to do this for the last chunk ends
        # - so that the below score will not 
        #   take those values into account
        #  x x x |  0    0
        #  x x x |  0    0 
        #  0 0 0 |  0    0 

        if limit_c < chunk_size:
            score = tl.where(
                offs_j[None, :] < limit_c,
                score, 0.
            )

        if limit_r < BLOCK_R:
            score = tl.where(
                offs_i[:, None] < limit_r,
                score, 0.0
            )

        # we need to compute 
        # _dzScore = (dZ * score).sum(-1, keepdim=True) # see notes above
        # dY = score * (dZ - _dzScore)
        #    = dZScore - score * _dzScore
        dZScore = dZ * tl.exp(score)

        # load _dZScore
        dZS_vec = tl.load(
            dZS_r + offs_i * denom_stride_seq,
            mask=offs_i < limit_r,
            other=0.0,
        ).to(tl.float32)

        qt_mat = tl.load(
            (
                qc_mat_ptr 
                + offs_v[:, None] * q_stride_dim # NOTE: assumed same
                + offs_i[None, :] * q_stride_seq
            ),
            mask=(offs_i[None, :] < limit_r),
            other=0.0
        ).to(tl.float32)

        # dY (new)
        dY = dZScore - tl.exp(score) * dZS_vec[:, None]

        # dK_2
        # - score (and therefore dZScore) should have
        #   zeros in appropriate places
        acc += tl.dot(qt_mat, dY)

        dot_mat = tl.load(
            (
                dout_r 
                + offs_v[:, None] * do_stride_dim # NOTE: assumed same
                + offs_i[None, :] * do_stride_seq
            ),
            mask=(offs_i[None, :] < limit_r),
            other=0.0
        ).to(tl.float32)

        # dV
        # - score should have zeros in appropriate places
        acc2 += tl.dot(dot_mat, tl.exp(score))

        # dZ1
        # - NOTE: i need to rotate dY.. dont have a good 
        # way to do it
        dYrotate = tl.dot(
            tl.where(
                (offs_i[:, None] - offs_i[None, :]) == 1,
                1.0, 0.0
            ),
            dY
        )
        dZ1 = tl.where(
            (offs_i[:, None] > (offs_j[None, :] + offset)), 
            dZ1_boundary[None, :] - tl.cumsum(dYrotate, axis=-2),
            0.0
        )

        # Z2 = deltanet_relu2 * ds 
        # term = Z2.pow(2/3) - Z2
        #      = x^2 - x^3
        #      = x^2 (1 - x)
        # where x = affinity, and 0 <= x <= 1

        # so then dsrc is 
        #   dZ2 * dest * deletanet_relu2
        #       = x^3 / src
        # where 
        #   dZ2 = -dZ1 / (3 * term)
        # so:
        #   dsrc = - dZ1 * x^3 / (3 * src * term)
        #        = - dZ1 * x^3 / (3 * x^2 (1-x) * src)
        #        = - dZ1 * x / (3 * (1-x) * src)

        A = -dZ1 / 3 * tl.exp(
            tl.log(tl.clamp(affinity, 1e-4, 1.0))

            # this is not decay
            - tl.log(1.0 - tl.clamp(affinity, 0.0, 1.0 - 1e-6)) 

            # recall it had pow(1/3) applied
            - 3 * tl.log(tl.clamp(src_vec[None, :], 1e-4, 1.0)) 
        ) 

        # dsrc needs to be reduced over rows
        acc3 += tl.sum(A, axis=-2)
        # move pointer with row chunk
        queries_r += BLOCK_R * q_stride_seq
        keys_r += BLOCK_R * k_stride_seq
        dout_r += BLOCK_R * do_stride_seq
        dZS_r += BLOCK_R * dZS_stride_seq
        dest += BLOCK_R * dest_stride_seq
        denom_r += BLOCK_R * denom_stride_seq

        # handle the limit
        limit_r -= BLOCK_R

        # for the next boundary
        if (
            ((r + 1) < nR)
            and
            ((r + 1) % cfac == 0)
        ):
            # move to next chunk
            chunked_dZ1 += dz1_stride_chunk

            # if at the chunk boundary
            dZ1_boundary = tl.load(
                chunked_dZ1 + offs_j,
                mask=(
                    offs_j < limit_c
                ),
                other=0.0,
            ).to(tl.float32)
        else:
            # minus from prev boundary
            dZ1_boundary -= tl.sum(dY, axis=-2)


    # -  DONE WITH ROW CHUNK LOOPS - 

    tl.store(
        (
            res_dK2_c 
            + offs_v[:, None] * res_dK2_stride_dim
            + offs_j[None, :] * res_dK2_stride_seq
        ),
        acc,
        mask=(
            (offs_j[None, :] < limit_c)
        )
    )

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
            res_src_c + offs_j * res_dsrc_stride_seq
        ),
        acc3,
        mask=(
            (offs_j < limit_c)
        )
    )