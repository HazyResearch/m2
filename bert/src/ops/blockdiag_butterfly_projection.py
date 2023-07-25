import math

import torch
import torch.nn as nn

from einops import rearrange

from src.models.layers.blockdiag_butterfly_multiply import blockdiag_butterfly_multiply
# from src.ops.low_rank import low_rank_project


# Copied here so it's more self-contained
def low_rank_project(M, rank):
    """Supports batches of matrices as well.
    """
    U, S, Vt = torch.linalg.svd(M)
    S_sqrt = S[..., :rank].sqrt()
    U = U[..., :rank] * rearrange(S_sqrt, '... rank -> ... 1 rank')
    Vt = rearrange(S_sqrt, '... rank -> ... rank 1') * Vt[..., :rank, :]
    return U, Vt


def factors(n):
    return [(i, n // i) for i in range(1, math.floor(math.sqrt(n)) + 1) if n % i == 0]


def blockdiag_butterfly_project(M, sizes=None):
    """Only works for square matrices for now
    """
    m, n = M.shape
    if m != n:
        raise NotImplementedError('Only support square matrices')
    if sizes is None:
        # Find the factors that are closest to sqrt(n)
        sizes = factors(n)[-1]
        # Larger factor first is probably more efficient, idk
        sizes = (sizes[1], sizes[0])
    assert n == sizes[0] * sizes[1]
    M_permuted_batched = rearrange(M, '(p k) (r s) -> k r p s', k=sizes[1], r=sizes[0])
    U, Vt = low_rank_project(M_permuted_batched, rank=1)
    w1_bfly = rearrange(Vt, 'k r 1 s -> r k s')
    w2_bfly = rearrange(U, 'k r s 1 -> k s r')
    return w1_bfly, w2_bfly


class ButterflyFFT(nn.Module):

    def __init__(self, n, direction='fft', norm='ortho', sizes=None):
        super().__init__()
        eye = torch.eye(n, dtype=torch.complex128)
        assert direction in ['fft', 'ifft']
        transform = torch.fft.fft if direction == 'fft' else torch.fft.ifft
        dft = transform(eye, norm=norm).t()
        # Find the factors that are closest to sqrt(n)
        sizes = factors(n)[-1]
        # Larger factor first is probably more efficient, idk
        sizes = (sizes[1], sizes[0])
        self.register_buffer('perm', rearrange(torch.arange(n), '(i j) -> (j i)', j=sizes[0]))
        w1, w2 = blockdiag_butterfly_project(dft[:, self.perm], sizes=sizes)
        # Store parameters as real instead of complex to avoid issues with Adam / AdamW
        self.w1_bfly = nn.Parameter(torch.view_as_real(w1.cfloat()))
        self.w2_bfly = nn.Parameter(torch.view_as_real(w2.cfloat()))

    def forward(self, x):
        w1_bfly, w2_bfly = torch.view_as_complex(self.w1_bfly), torch.view_as_complex(self.w2_bfly)
        return blockdiag_butterfly_multiply(rearrange(x[..., self.perm], '... n -> (...) n'),
                                            w1_bfly, w2_bfly).reshape_as(x)


class ButterflyFFT2(nn.Module):

    def __init__(self, n1, n2, direction='fft', norm='ortho'):
        """Input will have shape (..., n1, n2)
        """
        super().__init__()
        self.fft1 = ButterflyFFT(n1, direction=direction, norm=norm)
        self.fft2 = ButterflyFFT(n2, direction=direction, norm=norm)

    def forward(self, x):
        out = rearrange(self.fft1(rearrange(x, '... n1 n2 -> ... n2 n1')), '... n2 n1 -> ... n1 n2')
        return self.fft2(out)
