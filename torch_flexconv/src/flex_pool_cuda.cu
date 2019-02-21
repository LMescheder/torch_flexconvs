#include "ATen/ATen.h"
#include "cub/cub.cuh"
#include <limits>

inline int up2(int len, int th) { return (len - 1) / th + 1; }


// Implementations
template <typename scalar_t>
__global__
void flex_pool_forward_kernel_cuda_impl(
    const int B, const int N, const int K, const int D,
    const scalar_t* features,
    const int* neighborhood,
    scalar_t* output, 
    int* argmax,
    scalar_t float_min_value)
{
    const int b = blockIdx.z;

    for (int d = blockIdx.y * blockDim.y + threadIdx.y; d < D;
        d += blockDim.y * gridDim.y) 
    {
        for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < N;
             n += blockDim.x * gridDim.x)
        {
            scalar_t best_value = float_min_value;
            int best_id = 0;

            const int current_flat = b * D * N + d * N + n;

            for (int k_ = 0; k_ < K; ++k_) 
            {
                const int other_global_id = neighborhood[b * K * N + k_ * N + n];
                const scalar_t v = features[b * D * N + d * N + other_global_id];

                if (best_value < v)
                {
                    best_id = other_global_id;
                    best_value = v;
                }
            }

            output[current_flat] = best_value;
            argmax[current_flat] = best_id;
        }
    }
}

template <typename scalar_t>
__global__
void flex_pool_backward_kernel_cuda_impl(
    const int B, const int N, const int K, const int D,
    const scalar_t* features, 
    const int* neighborhood,
    const scalar_t* topdiff,
    const int* argmax,
    scalar_t* grad_features) 
{
    const int b = blockIdx.z;

    for (int d = blockIdx.y * blockDim.y + threadIdx.y; d < D;
         d += blockDim.y * gridDim.y)
    {
        for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < N;
             n += blockDim.x * gridDim.x)
        {
            const int top_id_flat = b * D * N + d * N + n;
            const int argmax_id = argmax[top_id_flat];
            const int bottom_id_flat = b * D * N + d * N + argmax_id;

            // TODO(patwie): scattered write, yeah :-(
            atomicAdd(&grad_features[bottom_id_flat], topdiff[top_id_flat]);
        }
    }
}

// Interface
void flex_pool_forward_kernel_cuda(
    at::Tensor features,
    at::Tensor neighborhood,
    at::Tensor output,
    at::Tensor argmax)
{
    // get dimensions
    const int B = neighborhood.size(0);
    const int K = neighborhood.size(1);
    const int N = neighborhood.size(2);
    const int D = features.size(1);

    const int threads = 32;
    dim3 block(threads, threads, 1);
    dim3 grid(up2(N, threads), up2(D, threads), B);

    argmax.zero_();
    
    AT_DISPATCH_FLOATING_TYPES(
        features.type(), "flex_pool_forward_kernel_cuda", ([&] 
    {
        output.fill_(std::numeric_limits<scalar_t>::lowest());

        flex_pool_forward_kernel_cuda_impl<scalar_t><<<grid, block>>>(
            B, N, K, D,
            features.data<scalar_t>(),
            neighborhood.data<int>(),
            output.data<scalar_t>(),
            argmax.data<int>(),
            std::numeric_limits<scalar_t>::lowest());
    }));
}

void flex_pool_backward_kernel_cuda(
    at::Tensor features,
    at::Tensor neighborhood,
    at::Tensor topdiff,
    at::Tensor argmax,
    at::Tensor grad_features)
{
    // get dimensions
    const int B = neighborhood.size(0);
    const int K = neighborhood.size(1);
    const int N = neighborhood.size(2);
    const int D = features.size(1);

    const int threads = 32;
    dim3 block(threads, threads, 1);
    dim3 grid(up2(N, threads), up2(D, threads), B);

    grad_features.zero_();

    AT_DISPATCH_FLOATING_TYPES(
        features.type(), "flex_pool_backward_kernel_cuda", ([&]
    {
        flex_pool_backward_kernel_cuda_impl<scalar_t><<<grid, block>>>(
            B, N, K, D,
            features.data<scalar_t>(),
            neighborhood.data<int>(),
            topdiff.data<scalar_t>(),
            argmax.data<int>(),
            grad_features.data<scalar_t>());
    }));
}