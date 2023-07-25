import torch

from einops import rearrange


def low_rank_project(M, rank):
    """Supports batches of matrices as well.
    """
    U, S, Vt = torch.linalg.svd(M)
    S_sqrt = S[..., :rank].sqrt()
    U = U[..., :rank] * rearrange(S_sqrt, '... rank -> ... 1 rank')
    Vt = rearrange(S_sqrt, '... rank -> ... rank 1') * Vt[..., :rank, :]
    return U, Vt
