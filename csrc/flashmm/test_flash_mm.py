import torch
import torch.nn.functional as F
from einops import rearrange
import math

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from flashmm import mm_block_fwd, hyena_filter_fwd, exp_mod_in_place_fwd

def ref_mm_block(
    u,
    linear, out_linear,
    x1_s, x2_s, v_s,
    x1_s_bias, x2_s_bias, v_s_bias,
    k, k_resid, D, Du,
    dropout_mask,
    gelu, fft_size
):
    x1x2v = linear(u)
    H = x1x2v.shape[-1] // 3
    seqlen = x1x2v.shape[-2]

    x1x2v = x1x2v.transpose(-1, -2)
    x1x2v_c = torch.nn.functional.conv1d(
        x1x2v,
        torch.flip(torch.cat([x1_s, x2_s, v_s], dim=0), dims=(-1,)).unsqueeze(1), # torch.flip to match our short conv
        bias=torch.cat([x1_s_bias, x2_s_bias, v_s_bias], dim=0), padding=x1_s.shape[-1] - 1, groups=x1x2v.shape[1]
    )[..., :seqlen]

    x1 = x1x2v_c[:, :H, :]
    x2 = x1x2v_c[:, H:2*H, :]
    v = x1x2v_c[:, 2*H:, :]

    x1 = x1 * v
    
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    x1_f = torch.fft.rfft(x1.to(dtype=k.dtype), n=fft_size)
    y = torch.fft.irfft(x1_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
    # y.shape:  B H L

    out = y + x1 * D[None, :, None]

    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        out = (out * rearrange(dropout_mask, "b H -> b H 1")).to(dtype=x1.dtype)
    else:
        out = out.to(dtype=x1.dtype)

    out = out * x2

    u = u.transpose(-1, -2)
    u_f = torch.fft.rfft(u, n=fft_size)
    k_resid_f = torch.fft.rfft(k_resid, n=fft_size) / fft_size
    out = out + torch.fft.irfft(u_f * k_resid_f, n=fft_size, norm="forward")[..., :seqlen] + Du[None, :, None] * u

    out = out.transpose(-1, -2)

    return out_linear(out)

def ref_hyena_filter(
    z, sin_freq, eo_mat, eo_bias, oo1_mat, oo1_bias, oo2_mat,
    oo2_bias, oh_mat, t, deltas, shift
):
    out = torch.bmm(z, eo_mat) + eo_bias.unsqueeze(1)
    out = torch.sin(out * sin_freq)
    out = torch.bmm(out, oo1_mat) + oo1_bias.unsqueeze(1)
    out = torch.sin(out * sin_freq)
    out = torch.bmm(out, oo2_mat) + oo2_bias.unsqueeze(1)
    out = torch.sin(out * sin_freq)
    out = torch.bmm(out, oh_mat)
    out = out * torch.exp(-t * deltas.abs()) + shift

    return out

def fast_mm_block(
    u,
    linear, out_linear,
    x1_s, x2_s, v_s,
    x1_s_bias, x2_s_bias, v_s_bias,
    k, k_resid, D, Du,
    dropout_mask,
    gelu, fft_size
):
    x1x2v = linear(u)
    H = x1x2v.shape[-1] // 3
    x1, x2, v = x1x2v.split(H, dim=-1)
    x1 = x1.transpose(1, 2).contiguous()
    x2 = x2.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    u = u.transpose(1, 2).contiguous()

    k_f = torch.fft.rfft(k, n=fft_size)
    k_residual_f = torch.fft.rfft(k_resid, n=fft_size)
    out = mm_block_fwd(
        x1, x2, v,
        x1_s, x2_s, v_s,
        x1_s_bias, x2_s_bias, v_s_bias,
        k_f, None, u, k_residual_f, Du, D, dropout_mask, gelu, fft_size,
        False, False
    )
    
    out = out.transpose(1, 2)

    return out_linear(out)

def fast_hyena_filter(
    z, sin_freq, eo_mat, eo_bias, oo1_mat, oo1_bias, oo2_mat,
    oo2_bias, oh_mat, min_delay, max_delay, shift, reverse_vec
):
    k = hyena_filter_fwd(
        z, sin_freq, eo_mat, eo_bias, oo1_mat, oo1_bias, oo2_mat, oo2_bias, reverse_vec, None
    )
    k = torch.bmm(k, oh_mat)
    return exp_mod_in_place_fwd(k, reverse_vec, min_delay, max_delay, shift)

B = 64
H = 768
L = 128

fftsize = 2 * L
device = 'cuda'
repeats = 30
short_conv_width = 4
gelu = True

torch.manual_seed(19)
u = torch.randn(B, L, H, device=device)
in_linear = torch.nn.Linear(H, 3 * H).to(device=device)
out_linear = torch.nn.Linear(H, H).to(device=device)
short_filter = torch.nn.Conv1d(3 * H, 3 * H, kernel_size=short_conv_width, padding=short_conv_width - 1, groups=3 * H, device=device)
x1_s = short_filter.weight[:H, 0, :]
x2_s = short_filter.weight[H:2*H, 0, :]
v_s = short_filter.weight[2*H:3*H, 0, :]
x1_s_bias = torch.zeros(short_filter.bias[:H].shape).to(device=device)
x2_s_bias = torch.zeros(short_filter.bias[H:2*H].shape).to(device=device)
v_s_bias = torch.zeros(short_filter.bias[2*H:3*H].shape).to(device=device)

k = torch.randn(H, L, device=device)
k_resid = torch.randn(H, L, device=device)
D = torch.randn(H, device=device)
Du = torch.randn(H, device=device)

out_ref = ref_mm_block(u, in_linear, out_linear, x1_s, x2_s, v_s, x1_s_bias, x2_s_bias, v_s_bias, k, k_resid, D, Du, None, gelu, fftsize)
out = fast_mm_block(u, in_linear, out_linear, x1_s, x2_s, v_s, x1_s_bias, x2_s_bias, v_s_bias, k, k_resid, D, Du, None, gelu, fftsize)

diff = (out_ref - out).abs().flatten()
argmax_diff = diff.argmax()
print("max diff for mm block:", diff[argmax_diff])
print("average diff for mm block:", diff.mean())

order = 128
emb_dim = 5
min_delay = math.log(1e-2) / 1.5
max_delay = math.log(1e-2) / 0.3
shift = 0.

z = torch.randn(1, L, emb_dim, device=device) * .02
sin_freq = torch.randn(1, order, device=device) * .02
eo_mat = torch.randn(1, emb_dim, order, device=device) * .02
eo_bias = torch.randn(1, order, device=device) * .02
oo1_mat = torch.randn(1, order, order, device=device) * .02
oo1_bias = torch.randn(1, order, device=device) * .02
oo2_mat = torch.randn(1, order, order, device=device) * .02
oo2_bias = torch.randn(1, order, device=device) * .02
oh_mat = torch.randn(1, order, H, device=device) * .02
reverse_vec = torch.zeros(1, device=device, dtype=torch.int32)

deltas = torch.linspace(min_delay, max_delay, H, device=device)[None, None]
t = torch.linspace(0, 1, L, device=device)[None, :, None]

out_ref = ref_hyena_filter(
    z, sin_freq, eo_mat, eo_bias, oo1_mat, oo1_bias, oo2_mat,
    oo2_bias, oh_mat, t, deltas, shift)
out = fast_hyena_filter(
    z, sin_freq, eo_mat, eo_bias, oo1_mat, oo1_bias, oo2_mat,
    oo2_bias, oh_mat, min_delay, max_delay, shift, reverse_vec)

diff = (out_ref - out).abs().flatten()
argmax_diff = diff.argmax()
print("max diff:", diff[argmax_diff])
print("avg diff:", diff.mean())
