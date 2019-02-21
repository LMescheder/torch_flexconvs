#include <torch/torch.h>
#include <limits>
#include "flex_conv.h"

// Implementations
template <scalar_t>
void flex_pool_forward_kernel_cpu_impl(
    at::TensorAccessor<scalar_t, 3> features,
    at::TensorAccesor<int, 3> neighborhood,
    at::TensorAccessor<scalar_t, 3> output,
    at::TensorAccessor<int, 3> argmax)
{
    // get dimensions
    const int B = neighborhood.size(0);
    const int K = neighborhood.size(1);
    const int N = neighborhood.size(2);
    const int D = features.size(1);

    for (int b = 0; b < B; ++b)
    {
        for (int d = 0; d < D; ++d)
        {
            for (int n = 0; n < N; ++n)
            {
                // max in neighborhood
                for (int k_ = 0; k_ < K; ++k_)
                {
                    const int other_global_id = neighborhood[b][k_][n];
                    if (output[b][d][n] < features[b][d][other_global_id])
                    {
                        argmax[b][d][n] = other_global_id;
                        output[b][d][n] = features[b][d][other_global_id];
                    }
                }
            }
        }
    }
}

template <scalar_t>
void flex_pool_backward_kernel_cpu_impl(
    at::TensorAccessor<scalar_t, 3> features,
    at::TensorAccessor<int, 3> neighborhood,
    at::TensorAccessor<scalar_t, 3> topdiff,
    at::TensorAccessor<int, 3> argmax,
    at::TensorAccessor<scalar_t, 3> grad_features)
{
    // get dimensions
    const int B = neighborhood.size(0);
    const int K = neighborhood.size(1);
    const int N = neighborhood.size(2);
    const int D = features.size(1);

    for (int b = 0; b < B; ++b)
    {
        for (int d = 0; d < D; ++d)
        {
            for (int n = 0; n < N; ++n)
            {
                grad_features[b][d][argmax[b][d][n]] += topdiff[b][d][n];
            }
        }
    }
}


// Interface
void flex_pool_forward_kernel_cpu(
    at::Tensor features,
    at::Tensor neighborhood,
    at::Tensor output,
    at::Tensor argmax)
{
    argmax.zero_();

    AT_DISPATCH_FLOATING_TYPES(
    features.type(), "flex_pool_forward_kernel_cpu", ([&] {
        output.fill_(std::numeric_limits<scalar_t>::lowest());
    
        flex_pool_forward_kernel_cpu_impl<scalar_t>(
            features.accessor<scalar_t, 3>(),
            neighborhood.accessor<int, 3>(),
            output.accessor<scalar_t, 3>(),
            argmax.accessor<int, 3>(),
    }));
}

void flex_pool_backward_kernel_cpu(
    at::Tensor features,
    at::Tensor neighborhood,
    at::Tensor topdiff,
    at::Tensor argmax,
    at::Tensor grad_features)
{
    grad_features.zero_();

    AT_DISPATCH_FLOATING_TYPES(
    features.type(), "flex_pool_backward_kernel_cpu", ([&] {
        flex_pool_backward_kernel_cpu_impl<scalar_t>(
            features.accessor<scalar_t, 3>(),
            neighborhood.accessor<int, 3>(),
            topdiff.accessor<scalar_t, 3>(),
            argmax.accessor<int, 3>(),
            grad_features.accessor<float, 3>();
    }));
}