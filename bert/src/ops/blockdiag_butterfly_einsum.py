import torch

from einops import rearrange

from src.ops.low_rank import low_rank_project


def blockdiag_butterfly_multiply_einsum_simple(x, w1_bfly, w2_bfly):
    """
    Arguments:
        x: (batch, n)
        w1_bfly: (k, j, i), where k = n / i
        w2_bfly: (j, l, k)
    Outputs:
        out: (batch, m), where m = l * j
    """
    batch, n = x.shape
    k, j, i = w1_bfly.shape
    j1, l, k1 = w2_bfly.shape
    assert j1 == j
    assert k1 == k
    assert k * i == n

    x_reshaped = rearrange(x, 'b (k i) -> b k i', k=k)
    out = torch.einsum('b k i, k j i, j l k -> b l j', x_reshaped, w1_bfly, w2_bfly)
    return rearrange(out, 'b l j -> b (l j)')


def blockdiag_butterfly_project_einsum_simple(M, nblocks1, nblocks2):
    """
    Arguments:
        M: (m, n)
    Outputs:
        w1_bfly: (nblocks1, nblocks2, i)
        w2_bfly: (nblocks2, l, nblocks1)
    """
    m, n = M.shape
    k, j = nblocks1, nblocks2
    M_permuted_batched = rearrange(M, '(l j) (k i) -> k j l i', k=nblocks1, j=nblocks2)
    U, Vt = low_rank_project(M_permuted_batched, rank=1)
    w1_bfly = rearrange(Vt, 'k j 1 i -> k j i')
    w2_bfly = rearrange(U, 'k j l 1 -> j l k')
    return w1_bfly, w2_bfly


def blockdiag_butterfly_multiply_einsum(x, w1_bfly, w2_bfly, b2):
    """
    Arguments:
        x: (batch, n)
        w1_bfly: (k, (j * b1), i), where k = n / i
        w2_bfly: (j, (l * b2), (k b1))
    Outputs:
        out: (batch, m), where m = l * j * b2
    """
    batch, n = x.shape
    k, jb1, i = w1_bfly.shape
    j, lb2, kb1 = w2_bfly.shape
    b1 = jb1 // j
    assert jb1 == j * b1
    assert kb1 == k * b1
    assert k * i == n

    x_reshaped = rearrange(x, 'b (k i) -> b k i', k=k)
    w1_bfly = rearrange(w1_bfly, 'k (j b1) i -> k j b1 i', b1=b1)
    w2_bfly = rearrange(w2_bfly, 'j (l b2) (k b1) -> j l b2 k b1', b1=b1, b2=b2)
    # torch.einsum doesn't support indices named b1 or b2, so we map b1 -> y, b2 -> z
    out = torch.einsum('b k i, k j y i, j l z k y -> b l j z', x_reshaped, w1_bfly, w2_bfly)
    return rearrange(out, 'b l j b2 -> b (l j b2)')


def blockdiag_butterfly_project_einsum(M, nblocks1, nblocks2, b1, b2):
    """
    Arguments:
        M: (m, n)
    Outputs:
        w1_bfly: (nblocks1, nblocks2, i)
        w2_bfly: (nblocks2, l, nblocks1)
    """
    m, n = M.shape
    k, j = nblocks1, nblocks2
    M_permuted_batched = rearrange(M, '(l j b2) (k i) -> k j (l b2) i', k=nblocks1, j=nblocks2,
                                   b2=b2)
    U, Vt = low_rank_project(M_permuted_batched, rank=b1)
    w1_bfly = rearrange(Vt, 'k j b1 i -> k (j b1) i')
    w2_bfly = rearrange(U, 'k j lb2 b1 -> j lb2 (k b1)')
    return w1_bfly, w2_bfly


def blockdiag_butterfly_multiply_einsum_rank(x, w1_bfly, w2_bfly):
    """
    Arguments:
        x: (batch, n)
        w1_bfly: (k, (r * j), i), where k = n / i
        w2_bfly: (j, l, (k r))
    Outputs:
        out: (batch, m), where m = l * j
    """
    batch, n = x.shape
    k, jb1, i = w1_bfly.shape
    j, l, kb1 = w2_bfly.shape
    r = jb1 // j
    assert jb1 == j * r
    assert kb1 == k * r
    assert k * i == n

    x_reshaped = rearrange(x, 'b (k i) -> b k i', k=k)
    w1_bfly = rearrange(w1_bfly, 'k (r j) i -> k r j i', r=r)
    w2_bfly = rearrange(w2_bfly, 'j l (k r) -> j l k r', r=r)
    out = torch.einsum('b k i, k r j i, j l k r -> b l j', x_reshaped, w1_bfly, w2_bfly)
    return rearrange(out, 'b l j -> b (l j)')


def blockdiag_butterfly_project_einsum_rank(M, nblocks1, nblocks2, rank):
    """
    Arguments:
        M: (m, n)
    Outputs:
        w1_bfly: (nblocks1, r * nblocks2, i)
        w2_bfly: (nblocks2, l, nblocks1 * r)
    """
    m, n = M.shape
    k, j = nblocks1, nblocks2
    M_permuted_batched = rearrange(M, '(l j) (k i) -> k j l i', k=nblocks1, j=nblocks2)
    U, Vt = low_rank_project(M_permuted_batched, rank=rank)
    w1_bfly = rearrange(Vt, 'k j r i -> k (r j) i')
    w2_bfly = rearrange(U, 'k j l r -> j l (k r)')
    return w1_bfly, w2_bfly
