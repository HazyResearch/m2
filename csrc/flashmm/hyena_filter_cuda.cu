// Copyright (c) 2023 Dan Fu
// Fused CUDA kernel to compute the Hyena filter
// This has low util, but it keeps everything on GPU
// Useful for when the order of the intermediate layers is small

#include <torch/torch.h>

#include <stdio.h>
#include <cmath>

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>

#include <c10/cuda/CUDAException.h>  // For C10_CUDA_KERNEL_LAUNCH_CHECK

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

#define MAX_ORDER 128 // hack for the bwd pass

__device__ float sum(float *data, int size) {
    // simple sum script to try to reduce numerical error
    if (size == 2) {
        return data[0] + data[1];
    }
    for (int i = 0; i < size / 2; i ++) {
        data[i] += data[i + size / 2];
    }
    if (size % 2 == 1) {
        data[0] += data[size - 1];
    }
    return sum(data, size / 2);
}

template<int EMB_DIM, int ORDER, typename input_t, typename output_t=input_t>
__global__ void hyena_filter_fwd_kernel(
    const input_t *__restrict__ inputDataZ,
    const input_t *__restrict__ inputDataSinFreq,
    const input_t *__restrict__ inputDataEoMat,
    const input_t *__restrict__ inputDataEoBias,
    const input_t *__restrict__ inputDataOo1Mat,
    const input_t *__restrict__ inputDataOo1Bias,
    const input_t *__restrict__ inputDataOo2Mat,
    const input_t *__restrict__ inputDataOo2Bias,
    const int *__restrict__ inputDataReverse,
    output_t *__restrict__ outputData,
    int L
) {
    extern __shared__ float shared_mem[];

    unsigned int C_id = blockIdx.x;
    unsigned int L_id = blockIdx.y;
    bool reverse = inputDataReverse[C_id] == 1;
    unsigned int L_id_out = reverse ? (L - blockIdx.y) : (blockIdx.y);
    unsigned int order_id = threadIdx.x;

    using BlockLoad_eo = cub::BlockLoad<input_t, ORDER, EMB_DIM, cub::BLOCK_LOAD_STRIPED>;
    using BlockLoad_oo = cub::BlockLoad<input_t, ORDER, ORDER, cub::BLOCK_LOAD_STRIPED>;
    using BlockLoad_oh = cub::BlockLoad<input_t, ORDER, ORDER, cub::BLOCK_LOAD_DIRECT>;

    float z[EMB_DIM];
    float scratch[ORDER];
    input_t eo_vec[EMB_DIM];
    input_t oo_vec[ORDER];

    // offset into sin freq, all the biases
    unsigned int order_offset = C_id * ORDER + order_id;
    // offset into order x order matrices
    unsigned int oo_offset = C_id * ORDER * ORDER;

    float cur_bias = 0;
    float cur_val = 0;
    float my_freq = float(inputDataSinFreq[order_offset]);

    // load z's
    for (int i = 0; i < EMB_DIM; i++) {
        z[i] = float(inputDataZ[C_id * L * EMB_DIM + L_id * EMB_DIM + i]);
    }

    // if ((threadIdx.x == 1) && (blockIdx.x == 1)) {
    //     for (int i = 0; i < EMB_DIM; i++) {
    //         printf("%f ", z[i]);
    //     }
    //     printf("\n");
    // }

    // load eo_mat
    BlockLoad_eo().Load(inputDataEoMat + C_id * EMB_DIM * ORDER, eo_vec);

    // do eo_mat * z
    #pragma unroll
    for (int i = 0; i < EMB_DIM; i++) {
        scratch[i] = float(eo_vec[i]) * z[i];
    }

    cur_val = sum(scratch, EMB_DIM);

    // do bias
    cur_bias = float(inputDataEoBias[order_offset]);
    cur_val += cur_bias;
    
    // if ((threadIdx.x == 1) && (blockIdx.x == 1)) {
    //     printf("after eo cur_val %f\n", cur_val);
    // }

    // do activation
    cur_val = std::sin(cur_val * my_freq);

    // if ((threadIdx.x == 1) && (blockIdx.x == 1)) {
    //     printf("after act, eo cur_val %f\n", cur_val);
    // }

    // save it in shared memory
    shared_mem[order_id] = cur_val;
    __syncthreads();

    // if ((threadIdx.x == 1) && (blockIdx.x == 1)) {
    //     for (int i = 0; i < ORDER; i++) {
    //         printf("%f ", shared_mem[i]);
    //     }
    //     printf("\n");
    // }

    // load oo1_mat
    BlockLoad_oo().Load(inputDataOo1Mat + oo_offset, oo_vec);

    // do oo_mat * order
    #pragma unroll
    for (int i = 0; i < ORDER; i++) {
        scratch[i] = float(oo_vec[i]) * shared_mem[i];
    }
    cur_val = sum(scratch, ORDER);

    // if ((threadIdx.x == 1) && (blockIdx.x == 1)) {
    //     printf("after oo1 cur_val %f\n", cur_val);
    // }

    // do bias
    cur_bias = float(inputDataOo1Bias[order_offset]);
    cur_val += cur_bias;

    // if ((threadIdx.x == 1) && (blockIdx.x == 1)) {
    //     printf("after oo1, bias cur_val %f, cur_bias %f\n", cur_val, cur_bias);
    // }

    // do activation
    cur_val = std::sin(cur_val * my_freq);

    // if ((threadIdx.x == 1) && (blockIdx.x == 1)) {
    //     printf("after act, oo1 cur_val %f\n", cur_val);
    // }

    // save it in shared memory
    __syncthreads();
    shared_mem[order_id] = cur_val;
    __syncthreads();
    // if ((threadIdx.x == 1) && (blockIdx.x == 1)) {
    //     for (int i = 0; i < ORDER; i++) {
    //         printf("%f ", shared_mem[i]);
    //     }
    //     printf("\n");
    // }

    // load oo2_mat
    BlockLoad_oo().Load(inputDataOo2Mat + oo_offset, oo_vec);

    // do oo_mat * order
    #pragma unroll
    for (int i = 0; i < ORDER; i++) {
        scratch[i] = float(oo_vec[i]) * shared_mem[i];
    }
    cur_val = sum(scratch, ORDER);

    // do bias
    cur_bias = float(inputDataOo2Bias[order_offset]);
    cur_val += cur_bias;

    // do activation
    cur_val = std::sin(cur_val * my_freq);

    outputData[C_id * L * ORDER + L_id_out * ORDER + order_id] = output_t(cur_val);

}

template <typename input_t, typename output_t=input_t>
void hyena_filter_fwd_cuda(
    const input_t *z, const input_t *sin_freq, const input_t *eo_mat,
    const input_t *eo_bias, const input_t *oo1_mat, const input_t *oo1_bias,
    const input_t *oo2_mat, const input_t *oo2_bias, const int *reverse,
    output_t *out, int C, int L, int emb_dim, int order
) {
    assert (emb_dim == 5);
    dim3 grid(C, L);
    dim3 block(order);
    switch(order) {
        case 64:
            hyena_filter_fwd_kernel<5, 64, input_t, output_t><<<grid, block>>>(z, sin_freq, eo_mat, eo_bias, oo1_mat, oo1_bias, oo2_mat, oo2_bias, reverse, out, L);
            break;
        case 128:
            hyena_filter_fwd_kernel<5, 64, input_t, output_t><<<grid, block>>>(z, sin_freq, eo_mat, eo_bias, oo1_mat, oo1_bias, oo2_mat, oo2_bias, reverse, out, L);
            break;
        default:
            AT_ERROR("flash hyena filter forward not implemented for this filter order");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t>
__global__ void exp_mod_in_place_fwd_kernel(
    input_t *__restrict__ inputDataK,
    const int *__restrict__ inputDataReverse,
    float min_decay, float max_decay, float shift
) {
    unsigned int L = gridDim.y;
    unsigned int H = blockDim.x;
    unsigned int C_id = blockIdx.y;
    unsigned int L_id = blockIdx.x;
    unsigned int H_id = threadIdx.x;
    bool reverse = inputDataReverse[C_id] == 1;
    float L_frac = reverse ? (-1. + float(L_id) / float(L - 1)) : (-1. * float(L_id) / float(L - 1));
    float H_frac = (float(H_id) / float(H - 1)) * (max_decay - min_decay) + min_decay;

    unsigned int offset = C_id * L * H + L_id * H + H_id;
    float cur_val = float(inputDataK[offset]);

    // if ((threadIdx.x == 10) && (blockIdx.x == 0) && (blockIdx.y == 5)) {
    //     printf("H_id %d, L_id %d\n", H_id, L_id);
    //     // printf("k[L=%d, H=%d]: %f\n", blockIdx.x, threadIdx.x, cur_val);
    //     printf("L_frac %f, H_frac %f\n", H_frac, L_frac);
    //     printf("\n");

    //     printf("Recompute H_frac ");
    //     printf("min_decay %f, max_decay %f\n", min_decay, max_decay);
    //     for (int i = 0; i < 10; i++) {
    //         float new_H_frac = (float(i) / float(H - 1)) * (max_decay - min_decay) + min_decay;
    //         printf("%f ", new_H_frac);
    //     }
    //     printf("\n");
    // }

    cur_val = cur_val * std::exp(std::abs(H_frac) * L_frac + shift);

    // if ((threadIdx.x == 0) && (blockIdx.x == 10)) {
    //     printf("H_id %d, L_id %d\n", H_id, L_id);
    //     printf("k[L=%d, H=%d]: %f\n", blockIdx.x, threadIdx.x, cur_val);
    //     printf("\n");
    // }
    inputDataK[offset] = input_t(cur_val);
}

template <typename input_t>
void exp_mod_in_place_fwd_cuda(
    input_t *k, const int *reverse, int C, int L, int H, float min_decay, float max_decay, float shift
) {
    dim3 grid(L, C);
    dim3 block(H);
    auto kernel = &exp_mod_in_place_fwd_kernel<input_t>;
    kernel<<<grid, block>>>(k, reverse, min_decay, max_decay, shift);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void hyena_filter_fwd_cuda<float, float>(
    const float *z, const float *sin_freq, const float *eo_mat,
    const float *eo_bias, const float *oo1_mat, const float *oo1_bias,
    const float *oo2_mat, const float *oo2_bias, const int *reverse,
    float *out, int C, int L, int emb_dim, int order);

template void hyena_filter_fwd_cuda<float, at::Half>(
    const float *z, const float *sin_freq, const float *eo_mat,
    const float *eo_bias, const float *oo1_mat, const float *oo1_bias,
    const float *oo2_mat, const float *oo2_bias, const int *reverse,
    at::Half *out, int C, int L, int emb_dim, int order);

template void hyena_filter_fwd_cuda<at::Half, at::Half>(
    const at::Half *z, const at::Half *sin_freq, const at::Half *eo_mat,
    const at::Half *eo_bias, const at::Half *oo1_mat, const at::Half *oo1_bias,
    const at::Half *oo2_mat, const at::Half *oo2_bias, const int *reverse,
    at::Half *out, int C, int L, int emb_dim, int order);

template void hyena_filter_fwd_cuda<at::BFloat16, at::BFloat16>(
    const at::BFloat16 *z, const at::BFloat16 *sin_freq, const at::BFloat16 *eo_mat,
    const at::BFloat16 *eo_bias, const at::BFloat16 *oo1_mat, const at::BFloat16 *oo1_bias,
    const at::BFloat16 *oo2_mat, const at::BFloat16 *oo2_bias, const int *reverse,
    at::BFloat16 *out, int C, int L, int emb_dim, int order);

template void exp_mod_in_place_fwd_cuda<float>(
    float *k, const int *reverse, int C, int L, int H, float min_decay, float max_decay, float shift);

template void exp_mod_in_place_fwd_cuda<at::Half>(
    at::Half *k, const int *reverse, int C, int L, int H, float min_decay, float max_decay, float shift);

template void exp_mod_in_place_fwd_cuda<at::BFloat16>(
    at::BFloat16 *k, const int *reverse, int C, int L, int H, float min_decay, float max_decay, float shift);