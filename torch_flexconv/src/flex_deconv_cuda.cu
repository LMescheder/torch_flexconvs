
#include "ATen/ATen.h"
#include "cub/cub.cuh"
#include "cuda_utils.h"

inline int up2(int len, int th) { return (len - 1) / th + 1; }


// Implementations
template <typename scalar_t>
__global__
void flex_deconv_forward_kernel_cuda_impl(
    const int B, const int N, const int K,
    const int Dp, const int Din, const int Dout, 
    const scalar_t *positions,
    const scalar_t *features,
    const int *neighborhood,
    const scalar_t *theta,
    const scalar_t *bias, 
    scalar_t *output)
{
    const int b = blockIdx.z;

    for (int n = blockIdx.y * blockDim.y + threadIdx.y; n < N;
         n += blockDim.y * gridDim.y)
    {
        const int self_k = neighborhood[b * K * N + 0 * N + n];

        for (int k_ = 0; k_ < K; ++k_)
        {
            const int other_k = neighborhood[b * K * N + k_ * N + n];

            for (int dout = blockIdx.x * blockDim.x + threadIdx.x; dout < Dout;
                 dout += blockDim.x * gridDim.x)
            {
                for (int din = 0; din < Din; ++din)
                {
                    const scalar_t v = features[b * Din * N + din * N + self_k];
                    scalar_t W = bias[din * Dout + dout];

                    for (int dp = 0; dp < Dp; ++dp)
                    {
                        scalar_t delta = positions[b * Dp * N + dp * N + other_k] -
                                      positions[b * Dp * N + dp * N + self_k];
                        W += theta[dp * Din * Dout + din * Dout + dout] * delta;
                    }

                    scalar_t Wv = W * v;
                    atomicAdd(&output[b * Dout * N + dout * N + other_k], Wv);
                }
            }
        }
    }
}

template <typename scalar_t>
__global__
void flex_deconv_backward_kernel_cuda_impl(
    const int B, const int N, const int K,
    const int Dp, const int Din, const int Dout,
    const scalar_t *positions,
    const scalar_t *features,
    const int *neighborhood,
    const scalar_t *theta,
    const scalar_t *bias,
    const scalar_t *top_diff,
    scalar_t *grad_features,
    scalar_t *grad_theta,
    scalar_t *grad_bias)
{
    const int b = blockIdx.z;

    // Compute
    // ---------------------------------------------------------------
    for (int n = blockIdx.y * blockDim.y + threadIdx.y; n < N;
         n += blockDim.y * gridDim.y)
    {
        const int self_k = neighborhood[b * K * N + 0 * N + n];

        for (int k_ = 0; k_ < K; ++k_)
        {
            const int other_k = neighborhood[b * K * N + k_ * N + n];

            for (int dout = blockIdx.x * blockDim.x + threadIdx.x; dout < Dout;
                 dout += blockDim.x * gridDim.x)
            {
                for (int din = 0; din < Din; ++din)
                {
                    const scalar_t current_top_diff =
                        top_diff[b * Dout * N + dout * N + other_k];
                    const scalar_t v = features[b * Din * N + din * N + self_k];

                    // update bias
                    scalar_t bias_update = v * current_top_diff;
                    atomicAdd(&grad_bias[din * Dout + dout], bias_update);

                    scalar_t W = bias[din * Dout + dout];

                    // update theta
                    for (int dp = 0; dp < Dp; ++dp)
                    {
                        scalar_t delta = positions[b * Dp * N + dp * N + other_k] -
                                      positions[b * Dp * N + dp * N + self_k];
                        scalar_t theta_update = v * delta * current_top_diff;
                        atomicAdd(
                            &grad_theta[dp * Din * Dout + din * Dout + dout], theta_update);

                        W += theta[dp * Din * Dout + din * Dout + dout] * delta;
                    }

                    // update features
                    scalar_t feature_update = W * current_top_diff;
                    atomicAdd(
                        &grad_features[b * Din * N + din * N + self_k], feature_update);
                }
            }
        }
    }
}

// Interface

void flex_deconv_forward_kernel_cuda(
    at::Tensor features,
    at::Tensor theta,
    at::Tensor bias,
    at::Tensor neighborhood,
    at::Tensor positions,
    at::Tensor output)
{
    using NBtype = int;

    const int B = neighborhood.size(0);
    const int K = neighborhood.size(1);
    const int N = neighborhood.size(2);
    const int Dp = theta.size(0);
    const int Din = theta.size(1);
    const int Dout = theta.size(2);

    const int threads = 32;
    dim3 block(threads, threads, 1);
    dim3 grid(up2(Dout, threads), up2(N, threads), B);

    output.zero_();

    AT_DISPATCH_FLOATING_TYPES(
        features.type(), "flex_deconv_forward_kernel_cuda", ([&] {
            flex_deconv_forward_kernel_cuda_impl<scalar_t><<<grid, block>>>(
                B, N, K, Dp, Din, Dout,
                positions.data<scalar_t>(),
                features.data<scalar_t>(),
                neighborhood.data<int>(),
                theta.data<scalar_t>(),
                bias.data<scalar_t>(),
                output.data<scalar_t>());
        }));
}

void flex_deconv_backward_kernel_cuda(
    at::Tensor features,
    at::Tensor theta,
    at::Tensor bias,
    at::Tensor neighborhood,
    at::Tensor positions,
    at::Tensor topdiff,
    at::Tensor grad_features,
    at::Tensor grad_theta,
    at::Tensor grad_bias)
{
    using NBtype = int;

    // get dimensions
    const int B = neighborhood.size(0);
    const int K = neighborhood.size(1);
    const int N = neighborhood.size(2);
    const int Dp = theta.size(0);
    const int Din = theta.size(1);
    const int Dout = theta.size(2);

    const int threads = 32;
    dim3 block(threads, threads, 1);
    dim3 grid(up2(Dout, threads), up2(N, threads), B);

    grad_features.zero_();
    grad_theta.zero_();
    grad_bias.zero_();

    AT_DISPATCH_FLOATING_TYPES(
        features.type(), "flex_deconv_backward_kernel_cuda", ([&] {
            flex_deconv_backward_kernel_cuda_impl<scalar_t><<<grid, block>>>(
                B, N, K, Dp, Din, Dout,
                positions.data<scalar_t>(),
                features.data<scalar_t>(),
                neighborhood.data<int>(),
                theta.data<scalar_t>(),
                bias.data<scalar_t>(),
                topdiff.data<scalar_t>(),
                grad_features.data<scalar_t>(),
                grad_theta.data<scalar_t>(),
                grad_bias.data<scalar_t>()
            );
        }));
}
