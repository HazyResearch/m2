/*
 * Copyright (c) 2023 Dan Fu
 * Code based off of flash FFT conv: Copyright (c) 2022 Tri Dao, Dan Fu
 */

#include <torch/torch.h>

#include <stdio.h>
#include <cuda/std/complex>

#include <cufftdx.hpp>

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>

#include <c10/cuda/CUDAException.h>  // For C10_CUDA_KERNEL_LAUNCH_CHECK

#include "static_switch.h"
#include "twiddle.cuh"

// *************** FOR ERROR CHECKING *******************
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL
// *************** FOR ERROR CHECKING *******************

template<int N>
inline __device__ void gelu(float (&output)[N], const float (&input)[N]) {
    constexpr float kAlpha = M_SQRT1_2;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        output[i] = input[i] * 0.5 * (1 + erff(input[i] * kAlpha));
    }
}

// GeLU(input0) * input1
template<int N>
inline __device__ void geglu(float (&output)[N], const float (&input0)[N], const float (&input1)[N]) {
    constexpr float kAlpha = M_SQRT1_2;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        output[i] = input1 * (input0[i] * 0.5 * (1 + erff(input0[i] * kAlpha)));
    }
}

template<typename T>
__device__ c10::complex<T> pointwise_mul(const c10::complex<T> a, const c10::complex<T> b) {
    return c10::complex<T>(a.real_ * b.real_, a.imag_ * b.imag_);
}


inline __device__ void read_rrii(cufftdx::detail::complex<__half2> val, c10::complex<float> result [2]) {
    using cfloat_t = c10::complex<float>;
    result[0] = cfloat_t(__half2float(val.x.x), __half2float(val.y.x));
    result[1] = cfloat_t(__half2float(val.x.y), __half2float(val.y.y));
}

inline __device__ cufftdx::detail::complex<__half2> write_rrii(c10::complex<float> val [2]) {
    using complex_t = typename cufftdx::detail::complex<__half2>;
    return complex_t {
        __float22half2_rn(float2 {val[0].real(), val[1].real()}),
        __float22half2_rn(float2 {val[0].imag(), val[1].imag()}),
    };
}

// Implement a real FFT of size 2 * N by calling a complex FFT of size N.
// http://www.robinscheibler.org/2013/02/13/real-fft.html
template<typename FFT>
inline __device__ void rfft(c10::complex<float> (&thread_data)[FFT::elements_per_thread],
                            c10::complex<float> *shared_mem){
    using cfloat_t = typename c10::complex<float>;
    using complex_t = typename cufftdx::detail::complex<float>;
    constexpr int N = cufftdx::size_of<FFT>::value;
    constexpr int EPT = FFT::elements_per_thread;

    complex_t *smem_c = reinterpret_cast<complex_t *>(shared_mem);
    complex_t (&thread_data_fft)[EPT] = reinterpret_cast<complex_t (&)[EPT]>(thread_data);
    FFT().execute(thread_data_fft, smem_c);
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < EPT; ++i) {
        smem_c[threadIdx.x + FFT::stride * i] = thread_data_fft[i];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < EPT; ++i) {
        if ((threadIdx.x == 0) && (i == 0)) {
            cfloat_t smem_val = shared_mem[0];
            thread_data[i] = cfloat_t(smem_val.real_ + smem_val.imag_, smem_val.real_ - smem_val.imag_);
        } else {
            int index = threadIdx.x + FFT::stride * i;
            cfloat_t smem_val_0 = shared_mem[index], smem_val_1 = shared_mem[N - index];
            cfloat_t X_even = smem_val_0 + std::conj(smem_val_1);
            // constexpr cfloat_t j = cfloat_t(0.f, 1.f);
            // cfloat_t X_odd = -j * (smem_val_0 - std::conj(smem_val_1));
            // Algebraic simplification
            cfloat_t X_odd = cfloat_t(smem_val_0.imag_ + smem_val_1.imag_, -smem_val_0.real_ + smem_val_1.real_);
            // cfloat_t twiddle;
            // sincospif(-float(index) / N, reinterpret_cast<float *>(&twiddle) + 1,
            //           reinterpret_cast<float *>(&twiddle));
            // Reading from lookup table is faster than computing the twiddle
            int quadrant = i / (EPT / 4);
            cfloat_t twiddle = twiddle_from_lut<N * 2>(quadrant, index);
            thread_data[i] = (X_even + X_odd * twiddle) / 2;
        }
    }
}

// Implement a conjugate symmetric inverse FFT of size 2 * N by calling a complex iFFT of size N.
// http://www.robinscheibler.org/2013/02/13/real-fft.html
template<typename IFFT>
inline __device__ void irfft(c10::complex<float> (&thread_data)[IFFT::elements_per_thread],
                             c10::complex<float> *shared_mem){
    using cfloat_t = typename c10::complex<float>;
    using complex_t = typename cufftdx::detail::complex<float>;
    constexpr int N = cufftdx::size_of<IFFT>::value;
    constexpr int EPT = IFFT::elements_per_thread;

    #pragma unroll
    for (int i = 0; i < EPT; ++i) {
        shared_mem[threadIdx.x + IFFT::stride * i] = thread_data[i];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < EPT; ++i) {
        if ((threadIdx.x == 0) && (i == 0)) {
            cfloat_t smem_val = shared_mem[0];
            thread_data[i] = cfloat_t(smem_val.real_ + smem_val.imag_, smem_val.real_ - smem_val.imag_);
            // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
            //     printf("%f.4f+%.4fi, ", thread_data[i].real_, thread_data[i].imag_);
            // }
        } else {
            int index = threadIdx.x + IFFT::stride * i;
            cfloat_t smem_val_0 = shared_mem[index], smem_val_1 = shared_mem[N - index];
            cfloat_t X_even = smem_val_0 + std::conj(smem_val_1);
            // cfloat_t twiddle;
            // sincospif(float(index) / N, reinterpret_cast<float *>(&twiddle) + 1,
            //           reinterpret_cast<float *>(&twiddle));;
            // Reading from lookup table is faster than computing the twiddle
            int quadrant = i / (EPT / 4);
            cfloat_t twiddle = std::conj(twiddle_from_lut<N * 2>(quadrant, index));
            // cfloat_t X_odd = (smem_val_0 - std::conj(smem_val_1)) * twiddle;
            // constexpr cfloat_t j = cfloat_t(0.f, 1.f);
            // thread_data[i] = (X_even + j * X_odd) / 2;
            // Algebraic simplification
            cfloat_t X_odd_j = cfloat_t(-smem_val_0.imag_ - smem_val_1.imag_, smem_val_0.real_ - smem_val_1.real_) * twiddle;
            thread_data[i] = X_even + X_odd_j;
        }
    }
    __syncthreads();
    IFFT().execute(reinterpret_cast<complex_t (&)[EPT]>(thread_data),
                   reinterpret_cast<complex_t *>(shared_mem));
}


/*
To dos: Take FFT of filter inside the kernel
*/
template<typename FFT, typename IFFT, typename input_t, typename output_t=input_t, bool GELU_OUTPUT=true>
__launch_bounds__( FFT::max_threads_per_block )
__global__ void mm_fwd_kernel(
    const input_t *__restrict__ inputDataX1,
    const input_t *__restrict__ inputDataX2,
    const input_t *__restrict__ inputDataV,
    const float *__restrict__ inputDataX1S,
    const float *__restrict__ inputDataX2S,
    const float *__restrict__ inputDataVS,
    const float *__restrict__ inputDataX1SBias,
    const float *__restrict__ inputDataX2SBias,
    const float *__restrict__ inputDataVSBias,
    const c10::complex<float> *__restrict__ filterData,
    const input_t *__restrict__ inputDataU,
    const c10::complex<float> *__restrict__ filterUData,
    const float *__restrict__ DuData,
    const float *__restrict__ filterTime,
    int filterLen,
    const float *__restrict__ DData,
    const float *__restrict__ dropmaskData,
    output_t *__restrict__ outputData,
    int batch_size,
    int H,
    int signal_size,
    int short_conv_width,
    size_t batch_stride, size_t H_stride,
    bool output_hbl_layout
) {

    using complex_t = typename cufftdx::detail::complex<float>;
    using cfloat_t = typename c10::complex<float>;
    constexpr int N = cufftdx::size_of<FFT>::value;
    constexpr int EPT = FFT::elements_per_thread;
    static_assert(FFT::storage_size == EPT);
    static_assert(IFFT::storage_size == EPT);

    using BlockLoad_input = cub::BlockLoad<cfloat_t, FFT::block_dim.x, EPT / 2, cub::BLOCK_LOAD_STRIPED>;
    using BlockLoad_filter = cub::BlockLoad<cfloat_t, FFT::block_dim.x, EPT, cub::BLOCK_LOAD_STRIPED>;
    using BlockStore_output = cub::BlockStore<c10::complex<output_t>, FFT::block_dim.x, EPT / 2, cub::BLOCK_STORE_STRIPED>;

    extern __shared__ cfloat_t shared_mem[];

    float result_data[EPT] = { 0 };

    cfloat_t filter_data[EPT];
    float x1_og_data[EPT];
    unsigned int filter_id = blockIdx.y;

    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("Before any loading\n");
    // }

    if (filterTime == nullptr) {
        BlockLoad_filter().Load(filterData + filter_id * (N + 1), filter_data);
        // CHECK THIS!!!
        if (threadIdx.x == 0) {
            filter_data[0].imag_ = *(reinterpret_cast<const float *>(filterData + filter_id * (N + 1) + N));
        }
        #pragma unroll
        for ( int i = 0; i < EPT; i++ ) { filter_data[i] /= 2 * N; }
    } else {
        // // load filter time into x1_og_data
        // BlockLoad_input().Load(
        //     reinterpret_cast<const c10::complex<input_t> *>(filterTime + filter_id * filterLen),
        //     reinterpret_cast<cfloat_t (&)[EPT / 2]>(x1_og_data),
        //     filterLen, cfloat_t(0.f));

        // // FFT(filter)
        // #pragma unroll
        // for (int i = 0; i < EPT; ++i) {
        //     filter_data[i] = i < EPT / 2 ? cfloat_t(x1_og_data[i * 2], x1_og_data[i * 2 + 1]) : cfloat_t(0.f);
        // }

        // // Execute FFT
        // rfft<FFT>(filter_data, shared_mem);
    }

    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("After loading filter\n");
    //     for (int i = 0; i < FFT::storage_size / 2; i++) {
    //         printf("%.4f+%.4fi, ", filter_data[i].real_, filter_data[i].imag_);
    //     }
    //     printf("\n");
    // }

    // CHECK THIS!!!
    float D_val = DData[filter_id];
    unsigned int dropmask_id = blockIdx.x * H + blockIdx.y;
    float dropmask_val = dropmaskData == nullptr ? 1.f : dropmaskData[dropmask_id];

    // Local array and copy data into it
    cfloat_t thread_data[EPT];

    // Id for inputData and inputMulQData
    size_t x1_offset = blockIdx.x * batch_stride + blockIdx.y * H_stride;

    BlockLoad_input().Load(
        reinterpret_cast<const c10::complex<input_t> *>(inputDataX1 + x1_offset),
        reinterpret_cast<cfloat_t (&)[EPT / 2]>(x1_og_data),
        signal_size / 2, cfloat_t(0.f));

    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("x1_data: ");
    //     for (int i = 0; i < EPT; i++) {
    //         printf("%.4f, ", x1_og_data[i]);
    //     }
    //     printf("\n");
    // }

    /** Compute short conv of x1 and x1 s **/
    
    // Load x1 s
    float short_conv_data[EPT];
    assert (short_conv_width <= EPT);

    size_t short_conv_offset = blockIdx.y;

    // read short conv data into registers
    for ( int i = 0; i < short_conv_width; i++ ) {
        short_conv_data[i] = float(inputDataX1S[short_conv_offset * short_conv_width + i]);
    }

    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("short_conv_data: ");
    //     for (int i = 0; i < short_conv_width; i++) {
    //         printf("%.4f, ", short_conv_data[i]);
    //     }
    //     printf("\n");
    // }

    // syncthreads, put x into shared memory
    __syncthreads();

    size_t num_threads = blockDim.x;
    size_t thread_id = threadIdx.x;
    size_t cur_L_idx_cfloat = 0;
    #pragma unroll
    for ( int i = 0; i < EPT / 2; i++ ) {
        cur_L_idx_cfloat = thread_id + i * num_threads + short_conv_width / 2;
        shared_mem[cur_L_idx_cfloat] = cfloat_t(x1_og_data[2 * i], x1_og_data[2 * i+1]);
    }
    if (thread_id < short_conv_width / 2) {
        shared_mem[thread_id] = cfloat_t(0.f, 0.f);
    }

    // syncthreads, compute convolution, put result into registers
    __syncthreads();

    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("x in shared memory: ");
    //     for (int i = 0; i < signal_size / 2; i++) {
    //         printf("%.4f, %.4f, ", shared_mem[i].real_, shared_mem[i].imag_);
    //     }
    //     printf("\n");
    // }

    #pragma unroll
    for ( int i = 0; i < EPT / 2; i++ ) {
        cur_L_idx_cfloat = thread_id + i * num_threads + short_conv_width / 2;
        x1_og_data[2 * i] = 0;
        x1_og_data[2 * i + 1] = 0;
        for ( int j = 0; j < short_conv_width / 2; j++ ) {
            x1_og_data[2 * i] += short_conv_data[2 * j] * shared_mem[cur_L_idx_cfloat - j].real_;
            x1_og_data[2 * i] += short_conv_data[2 * j + 1] * shared_mem[cur_L_idx_cfloat - j - 1].imag_;

            x1_og_data[2 * i + 1] += short_conv_data[2 * j] * shared_mem[cur_L_idx_cfloat - j].imag_;
            x1_og_data[2 * i + 1] += short_conv_data[2 * j + 1] * shared_mem[cur_L_idx_cfloat - j].real_;
        }
    }

    float bias = float(inputDataX1SBias[blockIdx.y]);

    // put it back in x1 and add bias
    #pragma unroll
    for ( int i = 0; i < EPT; i++ ) {
        // x1_og_data[i] = reinterpret_cast<float (&)[EPT * 2]>(thread_data)[i] + bias;
        x1_og_data[i] = x1_og_data[i] + bias;
    }

    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("bias: %.4f\n", bias);
    //     printf("x1_data: ");
    //     for (int i = 0; i < EPT; i++) {
    //         printf("%.4f, ", x1_og_data[i]);
    //     }
    //     printf("\n");
    // }

    /** Load V **/
    float v_data[EPT];
    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    size_t v_offset = blockIdx.x * batch_stride + (blockIdx.y) * H_stride;

    BlockLoad_input().Load(
        reinterpret_cast<const c10::complex<input_t> *>(inputDataV + v_offset),
        reinterpret_cast<cfloat_t (&)[EPT / 2]>(v_data),
        signal_size / 2, cfloat_t(0.f));
    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("v_data: ");
    //     for (int i = 0; i < EPT; i++) {
    //         printf("%.4f, ", v_data[i]);
    //     }
    //     printf("\n");
    // }

    /** Compute short conv over v **/
    
    // read short conv data into registers
    for ( int i = 0; i < short_conv_width; i++ ) {
        short_conv_data[i] = float(inputDataVS[short_conv_offset * short_conv_width + i]);
    }

    // syncthreads, put x into shared memory
    __syncthreads();

    #pragma unroll
    for ( int i = 0; i < EPT / 2; i++ ) {
        cur_L_idx_cfloat = thread_id + i * num_threads + short_conv_width / 2;
        shared_mem[cur_L_idx_cfloat] = cfloat_t(v_data[2 * i], v_data[2 * i+1]);
    }
    if (thread_id < short_conv_width / 2) {
        shared_mem[thread_id] = cfloat_t(0.f, 0.f);
    }

    // syncthreads, compute convolution, put result into registers
    __syncthreads();

    #pragma unroll
    for ( int i = 0; i < EPT / 2; i++ ) {
        cur_L_idx_cfloat = thread_id + i * num_threads + short_conv_width / 2;
        v_data[2 * i] = 0;
        v_data[2 * i + 1] = 0;
        for ( int j = 0; j < short_conv_width / 2; j++ ) {
            v_data[2 * i] += short_conv_data[2 * j] * shared_mem[cur_L_idx_cfloat - j].real_;
            v_data[2 * i] += short_conv_data[2 * j + 1] * shared_mem[cur_L_idx_cfloat - j - 1].imag_;

            v_data[2 * i + 1] += short_conv_data[2 * j] * shared_mem[cur_L_idx_cfloat - j].imag_;
            v_data[2 * i + 1] += short_conv_data[2 * j + 1] * shared_mem[cur_L_idx_cfloat - j].real_;
        }
    }

    bias = float(inputDataVSBias[blockIdx.y]);

    // put it back in v and add bias
    #pragma unroll
    for ( int i = 0; i < EPT; i++ ) {
        v_data[i] = v_data[i] + bias;
    }

    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("v_data: ");
    //     for (int i = 0; i < EPT; i++) {
    //         printf("%.4f, ", v_data[i]);
    //     }
    //     printf("\n");
    // }
    
    // x1 = x1 * v
    #pragma unroll
    for (int i = 0; i < EPT; ++i) {
        x1_og_data[i] *= v_data[i];
    }

    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("x1_og_data: ");
    //     for (int i = 0; i < EPT; i++) {
    //         printf("%.4f, ", x1_og_data[i]);
    //     }
    //     printf("\n");
    // }

    #pragma unroll
    for (int i = 0; i < EPT; ++i) {
        thread_data[i] = i < EPT / 2 ? cfloat_t(x1_og_data[i * 2], x1_og_data[i * 2 + 1]) : cfloat_t(0.f);
    }

    // Execute FFT
    __syncthreads();
    rfft<FFT>(thread_data, shared_mem);

    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("FFT(x1 * v)\n");
    //     for (int i = 0; i < EPT; i++) {
    //         printf("%.4f+%.4fi, ", thread_data[i].real_, thread_data[i].imag_);
    //     }
    //     printf("\n");
    // }

    #pragma unroll
    for ( int i = 0; i < EPT; i++ ) {
        thread_data[i] = (threadIdx.x == 0) && (i == 0) ?
            pointwise_mul(thread_data[i], filter_data[i]) : thread_data[i] * filter_data[i];
    }

    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("k_f * FFT(x1 * v)\n");
    //     for (int i = 0; i < EPT; i++) {
    //         printf("%.4f+%.4fi, ", thread_data[i].real_, thread_data[i].imag_);
    //     }
    //     printf("\n");
    // }

    // Execute FFT
    __syncthreads();
    irfft<IFFT>(thread_data, shared_mem);

    float out_data[EPT] {};

    #pragma unroll
    for ( int i = 0; i < EPT; i++ ) {
        out_data[i] = reinterpret_cast<float (&)[EPT * 2]>(thread_data)[i] + x1_og_data[i] * D_val;
    }

    // GELU_OUTPUT and dropout
    // https://github.com/pytorch/pytorch/blob/dc169d53aa266560750ea25ee0cf31c7e614550d/aten/src/ATen/native/cuda/Activation.cu#L395
    if (GELU_OUTPUT) { gelu(out_data, out_data); }
    #pragma unroll
    for ( int i = 0; i < EPT; i++ ) {
        out_data[i] *= dropmask_val;
    }

    float x2_data[EPT];

    BlockLoad_input().Load(
        reinterpret_cast<const c10::complex<input_t> *>(inputDataX2 + x1_offset),
        reinterpret_cast<cfloat_t (&)[EPT / 2]>(x2_data),
        signal_size / 2, cfloat_t(0.f));

    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("x2_data: ");
    //     for (int i = 0; i < EPT; i++) {
    //         printf("%.4f, ", x2_data[i]);
    //     }
    //     printf("\n");
    // }

    /** Compute short conv over x2 **/
   
    // read short conv data into registers
    for ( int i = 0; i < short_conv_width; i++ ) {
        short_conv_data[i] = float(inputDataX2S[short_conv_offset * short_conv_width + i]);
    }

    // syncthreads, put x into shared memory
    __syncthreads();

    #pragma unroll
    for ( int i = 0; i < EPT / 2; i++ ) {
        cur_L_idx_cfloat = thread_id + i * num_threads + short_conv_width / 2;
        shared_mem[cur_L_idx_cfloat] = cfloat_t(x2_data[2 * i], x2_data[2 * i+1]);
    }
    if (thread_id < short_conv_width / 2) {
        shared_mem[thread_id] = cfloat_t(0.f, 0.f);
    }

    // syncthreads, compute convolution, put result into registers
    __syncthreads();

    #pragma unroll
    for ( int i = 0; i < EPT / 2; i++ ) {
        cur_L_idx_cfloat = thread_id + i * num_threads + short_conv_width / 2;
        x2_data[2 * i] = 0;
        x2_data[2 * i + 1] = 0;
        for ( int j = 0; j < short_conv_width / 2; j++ ) {
            x2_data[2 * i] += short_conv_data[2 * j] * shared_mem[cur_L_idx_cfloat - j].real_;
            x2_data[2 * i] += short_conv_data[2 * j + 1] * shared_mem[cur_L_idx_cfloat - j - 1].imag_;

            x2_data[2 * i + 1] += short_conv_data[2 * j] * shared_mem[cur_L_idx_cfloat - j].imag_;
            x2_data[2 * i + 1] += short_conv_data[2 * j + 1] * shared_mem[cur_L_idx_cfloat - j].real_;
        }
    }

    bias = float(inputDataX2SBias[blockIdx.y]);

    // put it back in and add bias
    #pragma unroll
    for ( int i = 0; i < EPT; i++ ) {
        x2_data[i] = x2_data[i] + bias;
    }

    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("x2_data: ");
    //     for (int i = 0; i < EPT; i++) {
    //         printf("%.4f, ", x2_data[i]);
    //     }
    //     printf("\n");
    // }
    
    // out *= x2
    #pragma unroll
    for (int i = 0; i < EPT; ++i) {
        out_data[i] *= x2_data[i];
    }

    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("out_data: ");
    //     for (int i = 0; i < EPT; i++) {
    //         printf("%.4f, ", out_data[i]);
    //     }
    //     printf("\n");
    // }

    if (inputDataU != nullptr)  {
        BlockLoad_filter().Load(filterUData + filter_id * (N + 1), filter_data);
        // CHECK THIS!!!
        if (threadIdx.x == 0) {
            filter_data[0].imag_ = *(reinterpret_cast<const float *>(filterUData + filter_id * (N + 1) + N));
        }
        #pragma unroll
        for ( int i = 0; i < EPT; i++ ) { filter_data[i] /= 2 * N; }

        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        //     printf("Loading filter_u\n");
        //     for (int i = 0; i < FFT::storage_size / 2; i++) {
        //         printf("%.4f+%.4fi, ", filter_data[i].real_, filter_data[i].imag_);
        //     }
        //     printf("\n");
        // }

        BlockLoad_input().Load(
            reinterpret_cast<const c10::complex<input_t> *>(inputDataU + x1_offset),
            reinterpret_cast<cfloat_t (&)[EPT / 2]>(x1_og_data),
            signal_size / 2, cfloat_t(0.f));

        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        //     printf("x1_og_data: ");
        //     for (int i = 0; i < EPT; i++) {
        //         printf("%.4f, ", x1_og_data[i]);
        //     }
        //     printf("\n");
        // }

        #pragma unroll
        for (int i = 0; i < EPT; ++i) {
            thread_data[i] = i < EPT / 2 ? cfloat_t(x1_og_data[i * 2], x1_og_data[i * 2 + 1]) : cfloat_t(0.f);
        }
    
        // Execute FFT
        __syncthreads();
        rfft<FFT>(thread_data, shared_mem);

        #pragma unroll
        for ( int i = 0; i < EPT; i++ ) {
            thread_data[i] = (threadIdx.x == 0) && (i == 0) ?
                pointwise_mul(thread_data[i], filter_data[i]) : thread_data[i] * filter_data[i];
        }

        // Execute FFT
        __syncthreads();
        irfft<IFFT>(thread_data, shared_mem);
        
        D_val = DuData[filter_id];;
        #pragma unroll
        for ( int i = 0; i < EPT; i++ ) {
            out_data[i] += reinterpret_cast<float (&)[EPT * 2]>(thread_data)[i]  + x1_og_data[i] * D_val;
        }
    }

    #pragma unroll
    for (int i = 0; i < EPT; ++i) {
        result_data[i] += out_data[i];
        // result_data[i] = x1_og_data[i];
    }

    // Save results
    c10::complex<output_t> write_data[EPT / 2];
    #pragma unroll
    for (int i = 0; i < EPT / 2; ++i) {
        write_data[i] = c10::complex(output_t(result_data[i * 2]), output_t(result_data[i * 2 + 1]));
    }
    unsigned int output_fft_id = !output_hbl_layout ? blockIdx.x * H + blockIdx.y : blockIdx.x + blockIdx.y * batch_size;
    BlockStore_output().Store(reinterpret_cast<c10::complex<output_t> *>(outputData + output_fft_id * signal_size),
                              write_data, signal_size / 2);
    // TODO: what if signal_size is odd?
}

template <bool GELU_OUTPUT, uint FFT_SIZE, uint EPT, typename input_t, typename output_t=input_t>
void mm_fwd_cuda(
    const input_t *x1, const input_t *x2, const input_t *v,
    const float *x1_s, const float *x2_s, const float *v_s,
    const float *x1_s_bias, const float *x2_s_bias, const float *v_s_bias,
    const c10::complex<float> *filter,
    const input_t *u, const c10::complex<float> *u_filter, const float *Du,
    const float *filter_time, int filter_len,
    const float *D, const float *dropout_mask, output_t *out,
    int batch_size, int H, int signal_size, int short_conv_width,
    size_t batch_stride, size_t H_stride,
    bool output_hbl_layout
) {
#if defined(__CUDA_ARCH__)
    constexpr uint ARCH = __CUDA_ARCH__;
#else
    constexpr uint ARCH = 700;
#endif
    
    // (void) gelu_inp; // these options are not supported right now
    // (void) gelu_q;   // these options are not supported right now


    constexpr uint FPB = 1;
    // FFT is defined, its: size, type, direction, precision. Block() operator
    // informs that FFT will be executed on block level. Shared memory is
    // required for co-operation between threads.
    
    using FFT_base = decltype(cufftdx::Block() + cufftdx::Size<FFT_SIZE>() + cufftdx::Precision<float>() +
                            cufftdx::ElementsPerThread<EPT>() + cufftdx::FFTsPerBlock<FPB>() + cufftdx::SM<ARCH>()
                            + cufftdx::Type<cufftdx::fft_type::c2c>());

    using FFT = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::forward>());
    using IFFT = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::inverse>());

    // By default the shared memory size is 4 * FFT_SIZE (idk how).
    // So it wouldn't work for our rfft and irfft functions.
    const auto shared_memory_size = std::max({FFT::shared_memory_size, IFFT::shared_memory_size,
                                            8 * FFT_SIZE});
    // printf("shared_memory_size = %d\n", shared_memory_size);
    
    // unsigned int blocks_per_grid { static_cast<unsigned int>( std::ceil( batch_size / FPB ) ) };
    unsigned int H_per_grid { static_cast<unsigned int>( std::ceil( H / FPB ) ) };
    dim3 block(batch_size, H_per_grid);
    auto kernel = &mm_fwd_kernel<FFT, IFFT, input_t, output_t, GELU_OUTPUT>;
    // Increase dynamic memory limit if required.
    CUDA_RT_CALL( cudaFuncSetAttribute(kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        shared_memory_size ));
    kernel<<<block, FFT::block_dim, shared_memory_size>>>(
        x1, x2, v, x1_s, x2_s, v_s, x1_s_bias, x2_s_bias, v_s_bias,
        filter, u, u_filter, Du, filter_time, filter_len,
        D, dropout_mask, out, batch_size, H, signal_size,
        short_conv_width, batch_stride, H_stride, 
        output_hbl_layout);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
};

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
    size_t batch_stride, size_t H_stride, int fft_size,
    bool output_hbl_layout) {
    BOOL_SWITCH(gelu, GELU_OUTPUT, [&] {
        switch(fft_size) {
            case 256:
                mm_fwd_cuda<GELU_OUTPUT, 128, 4, input_t, output_t>(
                    x1, x2, v, x1_s, x2_s, v_s, x1_s_bias, x2_s_bias, v_s_bias,
                    filter, u, filter_u, Du, filter_time, filter_len,
                    D, dropout_mask, out, batch_size, H, signal_size,
                    short_conv_width, batch_stride, H_stride, 
                    output_hbl_layout);
                break;
            case 1024:
                mm_fwd_cuda<GELU_OUTPUT, 512, 16, input_t, output_t>(
                    x1, x2, v, x1_s, x2_s, v_s, x1_s_bias, x2_s_bias, v_s_bias,
                    filter, u, filter_u, Du, filter_time, filter_len,
                    D, dropout_mask, out, batch_size, H, signal_size,
                    short_conv_width, batch_stride, H_stride, 
                    output_hbl_layout);
                break;
            case 2048:
                mm_fwd_cuda<GELU_OUTPUT, 1024, 16, input_t, output_t>(
                    x1, x2, v, x1_s, x2_s, v_s, x1_s_bias, x2_s_bias, v_s_bias,
                    filter, u, filter_u, Du, filter_time, filter_len,
                    D, dropout_mask, out, batch_size, H, signal_size,
                    short_conv_width, batch_stride, H_stride, 
                    output_hbl_layout);
                break;
            case 4096:
                mm_fwd_cuda<GELU_OUTPUT, 2048, 8, input_t, output_t>(
                    x1, x2, v, x1_s, x2_s, v_s, x1_s_bias, x2_s_bias, v_s_bias,
                    filter, u, filter_u, Du, filter_time, filter_len,
                    D, dropout_mask, out, batch_size, H, signal_size,
                    short_conv_width, batch_stride, H_stride, 
                    output_hbl_layout);
                break;
            case 8192:
                mm_fwd_cuda<GELU_OUTPUT, 4096, 8, input_t, output_t>(
                    x1, x2, v, x1_s, x2_s, v_s, x1_s_bias, x2_s_bias, v_s_bias,
                    filter, u, filter_u, Du, filter_time, filter_len,
                    D, dropout_mask, out, batch_size, H, signal_size,
                    short_conv_width, batch_stride, H_stride, 
                    output_hbl_layout);
                break;
            case 16384:
                mm_fwd_cuda<GELU_OUTPUT, 8192, 8, input_t, output_t>(
                    x1, x2, v, x1_s, x2_s, v_s, x1_s_bias, x2_s_bias, v_s_bias,
                    filter, u, filter_u, Du, filter_time, filter_len,
                    D, dropout_mask, out, batch_size, H, signal_size,
                    short_conv_width, batch_stride, H_stride, 
                    output_hbl_layout);
                break;
            default:
                AT_ERROR("Monarch Mixer forward not implemented for this fft_size");
        }
    });
}

template void mm_fwd_cuda_dispatch<float, float>(
    const float *x1, const float *x2, const float *v,
    const float *x1_s, const float *x2_s, const float *v_s,
    const float *x1_s_bias, const float *x2_s_bias, const float *v_s_bias,
    const c10::complex<float> *filter,
    const float *u, const c10::complex<float> *filter_u, const float *Du,
    const float *filter_time, int filter_len,
    const float *D, const float *dropout_mask, float *out,
    bool gelu, int batch_size, int H, int signal_size, int short_conv_width,
    size_t batch_stride, size_t H_stride, int fft_size,
    bool output_hbl_layout);

template void mm_fwd_cuda_dispatch<float, at::Half>(
    const float *x1, const float *x2, const float *v,
    const float *x1_s, const float *x2_s, const float *v_s,
    const float *x1_s_bias, const float *x2_s_bias, const float *v_s_bias,
    const c10::complex<float> *filter,
    const float *u, const c10::complex<float> *filter_u, const float *Du,
    const float *filter_time, int filter_len,
    const float *D, const float *dropout_mask, at::Half *out,
    bool gelu, int batch_size, int H, int signal_size, int short_conv_width,
    size_t batch_stride, size_t H_stride, int fft_size,
    bool output_hbl_layout);

template void mm_fwd_cuda_dispatch<at::Half, at::Half>(
    const at::Half *x1, const at::Half *x2, const at::Half *v,
    const float *x1_s, const float *x2_s, const float *v_s,
    const float *x1_s_bias, const float *x2_s_bias, const float *v_s_bias,
    const c10::complex<float> *filter,
    const at::Half *u, const c10::complex<float> *filter_u, const float *Du,
    const float *filter_time, int filter_len,
    const float *D, const float *dropout_mask, at::Half *out,
    bool gelu, int batch_size, int H, int signal_size, int short_conv_width,
    size_t batch_stride, size_t H_stride, int fft_size,
    bool output_hbl_layout);

template void mm_fwd_cuda_dispatch<at::BFloat16, at::BFloat16>(
    const at::BFloat16 *x1, const at::BFloat16 *x2, const at::BFloat16 *v,
    const float *x1_s, const float *x2_s, const float *v_s,
    const float *x1_s_bias, const float *x2_s_bias, const float *v_s_bias,
    const c10::complex<float> *filter,
    const at::BFloat16 *u, const c10::complex<float> *filter_u, const float *Du,
    const float *filter_time, int filter_len,
    const float *D, const float *dropout_mask, at::BFloat16 *out,
    bool gelu, int batch_size, int H, int signal_size, int short_conv_width,
    size_t batch_stride, size_t H_stride, int fft_size,
    bool output_hbl_layout);
