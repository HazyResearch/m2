#include <vector>
#include <utility>
#include <cmath>
#include <torch/extension.h>

#include <cuda/std/complex>
#include <cuda_fp16.h>

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_FLOAT_AND_HALF_AND_BF16(INTYPE, OUTTYPE, NAME, ...)                     \
  if (INTYPE == at::ScalarType::Half) {                                                  \
    using input_t = at::Half;                                                            \
    using output_t = at::Half;                                                           \
    __VA_ARGS__();                                                                       \
  } else if (INTYPE == at::ScalarType::BFloat16) {                                       \
    using input_t = at::BFloat16;                                                        \
    using output_t = at::BFloat16;                                                       \
    __VA_ARGS__();                                                                       \
  } else if ((INTYPE == at::ScalarType::Float) && (OUTTYPE == at::ScalarType::Float))  { \
    using input_t = float;                                                               \
    using output_t = float;                                                              \
    __VA_ARGS__();                                                                       \
  } else if ((INTYPE == at::ScalarType::Float) && (OUTTYPE == at::ScalarType::Half))  {  \
    using input_t = float;                                                               \
    using output_t = at::Half;                                                           \
    __VA_ARGS__();                                                                       \
  } else {                                                                               \
    AT_ERROR(#NAME, " not implemented for in-type '", toString(INTYPE), "' and out-type '", toString(OUTTYPE), "'"); \
  }

#define DISPATCH_FLOAT_AND_HALF_AND_BF16_INPUT_ONLY(INTYPE, NAME, ...)                     \
  if (INTYPE == at::ScalarType::Half) {                                                  \
    using input_t = at::Half;                                                            \
    __VA_ARGS__();                                                                       \
  } else if (INTYPE == at::ScalarType::BFloat16) {                                       \
    using input_t = at::BFloat16;                                                        \
    __VA_ARGS__();                                                                       \
  } else if (INTYPE == at::ScalarType::Float)  { \
    using input_t = float;                                                               \
    __VA_ARGS__();                                                                       \
  } else {                                                                               \
    AT_ERROR(#NAME, " not implemented for in-type '", toString(INTYPE), "'"); \
  }

template <typename input_t, typename output_t=input_t>
void mm_fwd_cuda_dispatch(
    const input_t *x1, const input_t *x2, const input_t *v,
    const float *x1_s, const float *x2_s, const float *v_s,
    const float *x1_s_bias, const float *x2_s_bias, const float *v_s_bias,
    const c10::complex<float> *filter,
    const input_t *u, const c10::complex<float> *filter_u, const float *Du,
    const float *filter_time, int filter_len,
    const float *D, const float *dropout_mask, output_t *out,
    bool gelu, int batch_size, int H, int signal_size, int short_conv_width,
    size_t batch_stride, size_t H_stride, int fft_size, bool output_hbl_layout);

template <typename input_t, typename output_t=input_t>
void hyena_filter_fwd_cuda(
    const input_t *z, const input_t *sin_freq, const input_t *eo_mat,
    const input_t *eo_bias, const input_t *oo1_mat, const input_t *oo1_bias,
    const input_t *oo2_mat, const input_t *oo2_bias, const int *reverse,
    output_t *out, int C, int L, int emb_dim, int order);

template <typename input_t>
void exp_mod_in_place_fwd_cuda(
    input_t *k, const int *reverse, int C, int L, int H, float min_decay, float max_decay, float shift);

torch::Tensor mm_block_fwd(
  torch::Tensor x1,
  torch::Tensor x2,
  torch::Tensor v,
  torch::Tensor x1_s,
  torch::Tensor x2_s,
  torch::Tensor v_s,
  torch::Tensor x1_s_bias,
  torch::Tensor x2_s_bias,
  torch::Tensor v_s_bias,
  torch::Tensor filter_f,
  c10::optional<torch::Tensor> filter,
  c10::optional<torch::Tensor> u,
  c10::optional<torch::Tensor> filter_u_f,
  c10::optional<torch::Tensor> Du,
  torch::Tensor D,
  c10::optional<torch::Tensor> dropout_mask,
  bool gelu, int fft_size,
  bool force_fp16_output, bool output_hbl_layout
) {
    CHECK_DEVICE(x1);
    CHECK_DEVICE(x2);
    CHECK_DEVICE(v);
    CHECK_DEVICE(x1_s);
    CHECK_DEVICE(x2_s);
    CHECK_DEVICE(v_s);
    CHECK_DEVICE(x1_s_bias);
    CHECK_DEVICE(x2_s_bias);
    CHECK_DEVICE(v_s_bias);
    CHECK_DEVICE(filter_f);
    CHECK_DEVICE(D);

    TORCH_CHECK(x1.stride(-1) == 1);
    TORCH_CHECK(x2.stride(-1) == 1);
    TORCH_CHECK(v.stride(-1) == 1);

    TORCH_CHECK(x2.stride(0) == x1.stride(0) && x2.stride(1) == x1.stride(1));
    TORCH_CHECK(x2.dtype() == x1.dtype());
    TORCH_CHECK(v.stride(0) == x1.stride(0) && v.stride(1) == x1.stride(1));
    TORCH_CHECK(v.dtype() == x1.dtype());

    TORCH_CHECK(x1_s.is_contiguous());
    TORCH_CHECK(x2_s.is_contiguous());
    TORCH_CHECK(v_s.is_contiguous());
    TORCH_CHECK(filter_f.is_contiguous());
    TORCH_CHECK(D.is_contiguous());

    const int batch_size = x1.size(0);
    const int H = x2.size(1);
    const int L = x2.size(2);
    CHECK_SHAPE(x1, batch_size, H, L);
    CHECK_SHAPE(x2, batch_size, H, L);
    CHECK_SHAPE(v, batch_size, H, L);

    const int short_conv_width = x1_s.size(1);
    TORCH_CHECK(short_conv_width % 2 == 0);
    CHECK_SHAPE(x1_s, H, short_conv_width);
    CHECK_SHAPE(x2_s, H, short_conv_width);
    CHECK_SHAPE(v_s, H, short_conv_width);
    CHECK_SHAPE(x1_s_bias, H);
    CHECK_SHAPE(x2_s_bias, H);
    CHECK_SHAPE(v_s_bias, H);
    CHECK_SHAPE(filter_f, H, fft_size / 2 + 1);
    CHECK_SHAPE(D, H);

    if (filter.has_value()) {
      auto filter_value = filter.value();
      CHECK_DEVICE(filter_value);
      const int filter_len = filter_value.size(1);
      CHECK_SHAPE(filter_value, H, filter_len);
      TORCH_CHECK(filter_value.is_contiguous());
    }
    if (u.has_value()) {
      auto u_value = u.value();
      CHECK_DEVICE(u_value);
      CHECK_SHAPE(u_value, batch_size, H, L);
      TORCH_CHECK(u_value.is_contiguous());
      TORCH_CHECK(u_value.dtype() == x1.dtype());
      assert (filter_u_f.has_value());

      auto filter_u_f_value = filter_u_f.value();
      CHECK_DEVICE(filter_u_f_value);
      CHECK_SHAPE(filter_u_f_value, H, fft_size / 2 + 1);
      TORCH_CHECK(filter_u_f_value.is_contiguous());

      auto Du_value = Du.value();
      CHECK_DEVICE(Du_value);
      CHECK_SHAPE(Du_value, H);
      TORCH_CHECK(Du_value.dtype() == torch::kFloat32);
    }

    TORCH_CHECK(x1.dtype() == torch::kFloat16 || x1.dtype() == torch::kFloat32 || x1.dtype() == torch::kBFloat16);
    // TODO: check filter.dtype is complex64 (no complex32)
    TORCH_CHECK(x1_s.dtype() == torch::kFloat32);
    TORCH_CHECK(x2_s.dtype() == torch::kFloat32);
    TORCH_CHECK(v_s.dtype() == torch::kFloat32);
    TORCH_CHECK(x1_s_bias.dtype() == torch::kFloat32);
    TORCH_CHECK(x2_s_bias.dtype() == torch::kFloat32);
    TORCH_CHECK(v_s_bias.dtype() == torch::kFloat32);
    TORCH_CHECK(D.dtype() == torch::kFloat32);

    if (dropout_mask.has_value()) {
        auto dropout_mask_value = dropout_mask.value();
        CHECK_DEVICE(dropout_mask_value);
        CHECK_SHAPE(dropout_mask_value, batch_size, H);
        TORCH_CHECK(dropout_mask_value.dtype() == torch::kFloat32);
    }

    auto opts = x1.options();
    at::ScalarType x1_dtype = ::detail::scalar_type(x1.scalar_type());
    if (x1.dtype() == at::ScalarType::BFloat16) { force_fp16_output = false; }
    auto out = !output_hbl_layout
        ? torch::empty({batch_size, H, L}, opts.dtype(force_fp16_output ? torch::kFloat16 : x1_dtype))
        : torch::empty({H, batch_size, L}, opts.dtype(force_fp16_output ? torch::kFloat16 : x1_dtype)).permute({1, 0, 2});
    TORCH_CHECK((L <= fft_size / 2) && (L % 2 == 0));
    TORCH_CHECK(fft_size >= 16 && fft_size <= 16384 && (fft_size == 1 << int(log2(float(fft_size)))));

    size_t batch_stride = x1.stride(0), H_stride = x1.stride(1);
    DISPATCH_FLOAT_AND_HALF_AND_BF16(x1.scalar_type(), out.scalar_type(), "mm_block_fwd", [&] {
        mm_fwd_cuda_dispatch(
            // inputs
            static_cast<input_t *>(x1.data_ptr()),
            static_cast<input_t *>(x2.data_ptr()),
            static_cast<input_t *>(v.data_ptr()),

            // short convs
            static_cast<float *>(x1_s.data_ptr()),
            static_cast<float *>(x2_s.data_ptr()),
            static_cast<float *>(v_s.data_ptr()),
            static_cast<float *>(x1_s_bias.data_ptr()),
            static_cast<float *>(x2_s_bias.data_ptr()),
            static_cast<float *>(v_s_bias.data_ptr()),

            // long conv
            static_cast<c10::complex<float> *>(filter_f.data_ptr()),

            // residual connection
            u.has_value() ? static_cast<input_t *>(u.value().data_ptr()) : nullptr,
            filter_u_f.has_value() ? static_cast<c10::complex<float> *>(filter_u_f.value().data_ptr()) : nullptr,
            Du.has_value() ? static_cast<float *>(Du.value().data_ptr()) : nullptr,

            // filter by time
            filter.has_value() ? static_cast<float *>(filter.value().data_ptr()) : nullptr,
            filter.has_value() ? filter.value().size(1) : 0,

            // bias
            static_cast<float *>(D.data_ptr()),

            // other params
            dropout_mask.has_value() ? static_cast<float *>(dropout_mask.value().data_ptr()) : nullptr,
            static_cast<output_t *>(out.data_ptr()),
            gelu, batch_size, H, L, short_conv_width,
            batch_stride, H_stride, fft_size,
            output_hbl_layout);
    });
    return out;
}

torch::Tensor hyena_filter_fwd_preprocess(
    torch::Tensor z,
    torch::Tensor sin_freq,
    torch::Tensor eo_mat,
    torch::Tensor eo_bias,
    torch::Tensor oo1_mat,
    torch::Tensor oo1_bias,
    torch::Tensor oo2_mat,
    torch::Tensor oo2_bias,
    torch::Tensor reverse,
    c10::optional<torch::Tensor> out
) {
    CHECK_DEVICE(z);
    CHECK_DEVICE(sin_freq);
    CHECK_DEVICE(eo_mat);
    CHECK_DEVICE(eo_bias);
    CHECK_DEVICE(oo1_mat);
    CHECK_DEVICE(oo1_bias);
    CHECK_DEVICE(oo2_mat);
    CHECK_DEVICE(oo2_bias);
    // CHECK_DEVICE(oh_mat);

    const int C = z.size(0);
    const int L = z.size(1);
    const int emb_dim = z.size(2);
    const int order = eo_mat.size(-1);
    // const int H = oh_mat.size(-1);

    TORCH_CHECK(z.is_contiguous());
    TORCH_CHECK(eo_mat.is_contiguous());
    TORCH_CHECK(oo1_mat.is_contiguous());
    TORCH_CHECK(oo2_mat.is_contiguous());
    // TORCH_CHECK(oh_mat.is_contiguous());

    CHECK_SHAPE(z, C, L, emb_dim);
    CHECK_SHAPE(sin_freq, C, order);
    CHECK_SHAPE(eo_mat, C, emb_dim, order);
    CHECK_SHAPE(eo_bias, C, order);
    CHECK_SHAPE(oo1_mat, C, order, order);
    CHECK_SHAPE(oo1_bias, C, order);
    CHECK_SHAPE(oo2_mat, C, order, order);
    CHECK_SHAPE(oo2_bias, C, order);
    CHECK_SHAPE(reverse, C);
    // CHECK_SHAPE(oh_mat, H, order);

    TORCH_CHECK(z.dtype() == sin_freq.dtype());
    TORCH_CHECK(z.dtype() == eo_mat.dtype());
    TORCH_CHECK(z.dtype() == eo_bias.dtype());
    TORCH_CHECK(z.dtype() == oo1_mat.dtype());
    TORCH_CHECK(z.dtype() == oo1_bias.dtype());
    TORCH_CHECK(z.dtype() == oo2_mat.dtype());
    TORCH_CHECK(z.dtype() == oo2_bias.dtype());
    TORCH_CHECK(reverse.dtype() == at::ScalarType::Int);
    // TORCH_CHECK(z.dtype() == oh_mat.dtype());

    // TORCH_CHECK((H % order) == 0);

    auto opts = z.options();
    at::ScalarType z_dtype = ::detail::scalar_type(z.scalar_type());
    torch::Tensor out_vec;
    if (out.has_value()) {
      out_vec = out.value();
      CHECK_DEVICE(out_vec);
      TORCH_CHECK(out_vec.is_contiguous());
      CHECK_SHAPE(out_vec, C, L, order);
      TORCH_CHECK(out_vec.dtype() == z_dtype);
    } else {
      out_vec = torch::empty({C, L, order}, opts.dtype(z_dtype));
    }

    return out_vec;
}

torch::Tensor hyena_filter_fwd(
    torch::Tensor z,
    torch::Tensor sin_freq,
    torch::Tensor eo_mat,
    torch::Tensor eo_bias,
    torch::Tensor oo1_mat,
    torch::Tensor oo1_bias,
    torch::Tensor oo2_mat,
    torch::Tensor oo2_bias,
    torch::Tensor reverse,
    c10::optional<torch::Tensor> out
) {
    const int C = z.size(0);
    const int L = z.size(1);
    const int emb_dim = z.size(2);
    const int order = eo_mat.size(-1);

    torch::Tensor out_vec = hyena_filter_fwd_preprocess(
        z, sin_freq, eo_mat, eo_bias, oo1_mat, oo1_bias, oo2_mat, oo2_bias, reverse, out);

    DISPATCH_FLOAT_AND_HALF_AND_BF16(z.scalar_type(), out_vec.scalar_type(), "hyena_filter_fwd", [&] {
        hyena_filter_fwd_cuda(
            static_cast<input_t *>(z.data_ptr()),
            static_cast<input_t *>(sin_freq.data_ptr()),
            static_cast<input_t *>(eo_mat.data_ptr()),
            static_cast<input_t *>(eo_bias.data_ptr()),
            static_cast<input_t *>(oo1_mat.data_ptr()),
            static_cast<input_t *>(oo1_bias.data_ptr()),
            static_cast<input_t *>(oo2_mat.data_ptr()),
            static_cast<input_t *>(oo2_bias.data_ptr()),
            static_cast<int *>(reverse.data_ptr()),
            // static_cast<input_t *>(oh_mat.data_ptr()),
            static_cast<output_t *>(out_vec.data_ptr()),
            C, L, emb_dim, order
        );
    });
    return out_vec;
}

torch::Tensor exp_mod_in_place_fwd(
  torch::Tensor k,
  torch::Tensor reverse,
  float min_decay, float max_decay, float shift)
{
    CHECK_DEVICE(k);
    int C = k.size(0);
    int L = k.size(1);
    int H = k.size(2);
    CHECK_SHAPE(k, C, L, H);
    CHECK_SHAPE(reverse, C);
    TORCH_CHECK(k.is_contiguous());
    TORCH_CHECK(reverse.dtype() == at::ScalarType::Int);

    DISPATCH_FLOAT_AND_HALF_AND_BF16_INPUT_ONLY(k.scalar_type(), "exp_mod_in_place_fwd", [&] {
        exp_mod_in_place_fwd_cuda(
            static_cast<input_t *>(k.data_ptr()),
            static_cast<int *>(reverse.data_ptr()),
            C, L, H, min_decay, max_decay, shift
        );
    });
    return k;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mm_block_fwd", &mm_block_fwd, "Fused Monarch Mixer Block Kernel, forward");
    m.def("hyena_filter_fwd", &hyena_filter_fwd, "Fused Hyena Filter Kernel, forward");
    m.def("exp_mod_in_place_fwd", &exp_mod_in_place_fwd, "Fused exponential modulation, forward");
}
