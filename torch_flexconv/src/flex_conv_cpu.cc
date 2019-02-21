#include <torch/torch.h>
#include "flex_conv.h"


// Implementations
template <typename scalar_t>
void flex_conv_forward_kernel_cpu_impl(
    at::TensorAccessor<scalar_t, 3> features,
    at::TensorAccessor<scalar_t, 3> theta,
    at::TensorAccessor<scalar_t, 2> bias,
    at::TensorAccessor<int, 3> neighborhood,
    at::TensorAccessor<scalar_t, 3> positions,
    at::TensorAccessor<scalar_t, 3> output)
{
    // get dimensions
    const int B = neighborhood.size(0);
    const int K = neighborhood.size(1);
    const int N = neighborhood.size(2);
    const int Dp = theta.size(0);
    const int Din = theta.size(1);
    const int Dout = theta.size(2);


    for (int b = 0; b < B; ++b)
    {
        for (int n = 0; n < N; ++n)
        {
            for (int k_ = 0; k_ < K; ++k_)
            {
                int k = neighborhood[b][k_][n];
                for (int dout = 0; dout < Dout; ++dout)
                {
                    for (int din = 0; din < Din; ++din)
                    {
                        const scalar_t v = features[b][din][k];
                        scalar_t W = bias[din][dout];
                        for (int dp = 0; dp < Dp; ++dp)
                        {
                            scalar_t delta = positions[b][dp][k] -
                                             positions[b][dp][neighborhood[b][0][n]];
                            W += theta[dp][din][dout] * delta;
                        }
                        output[b][dout][n] = output[b][dout][n] + W * v;
                    }
                }
            }
        }
    }
}

template <typename scalar_t>
void flex_conv_backward_kernel_cpu_impl(
    at::TensorAccessor<scalar_t, 3> features,
    at::TensorAccessor<scalar_t, 3> theta,
    at::TensorAccessor<scalar_t, 2> bias,
    at::TensorAccessor<int, 3> neighborhood,
    at::TensorAccessor<scalar_t, 3> positions,
    at::TensorAccessor<scalar_t, 3> topdiff,
    at::TensorAccessor<scalar_t, 3> grad_features,
    at::TensorAccessor<scalar_t, 3> grad_theta,
    at::TensorAccessor<scalar_t, 2> grad_bias)
{
    // get dimensions
    const int B = neighborhood.size(0);
    const int K = neighborhood.size(1);
    const int N = neighborhood.size(2);
    const int Ddegree = theta.size(0);
    const int Dp = theta.size(0);
    const int Din = theta.size(1);
    const int Dout = theta.size(2);


    // ========================= bias ==============================
    for (int b = 0; b < B; ++b)
    {
        for (int n = 0; n < N; ++n)
        {
            for (int k_ = 0; k_ < K; ++k_)
            {
                int k = neighborhood[b][k_][n];

                for (int j = 0; j < Din; ++j)
                {
                    for (int l = 0; l < Dout; ++l)
                    {
                        grad_bias[j][l] += features[b][j][k] * topdiff[b][l][n];
                    }
                }
            }
        }
    }

    // ========================= theta ==============================
    for (int b = 0; b < B; ++b)
    {
        for (int n = 0; n < N; ++n)
        {
            for (int k_ = 0; k_ < K; ++k_)
            {
                int k = neighborhood[b][k_][n];

                for (int j = 0; j < Din; ++j)
                {
                    for (int l = 0; l < Dout; ++l)
                    {
                        for (int i = 0; i < Dp; ++i)
                        {
                            const scalar_t delta =
                                positions[b][i][k] - positions[b][i][neighborhood[b][0][n]];

                            grad_theta[i][j][l] +=
                                features[b][j][k] * delta * topdiff[b][l][n];
                        }
                    }
                }
            }
        }
    }

    // ========================= features ==============================
    for (int b = 0; b < B; ++b)
    {
        for (int n = 0; n < N; ++n)
        {
            for (int k_ = 0; k_ < K; ++k_)
            {
                int k = neighborhood[b][k_][n];

                for (int j = 0; j < Din; ++j) {
                    for (int l = 0; l < Dout; ++l)
                    {
                        scalar_t W = bias[j][l];
                        for (int i = 0; i < Dp; ++i)
                        {
                            const scalar_t delta =
                                positions[b][i][k] - positions[b][i][neighborhood[b][0][n]];

                            W += theta[i][j][l] * delta;
                        }
                        grad_features[b][j][k] += W * topdiff[b][l][n];
                    }
                }
            }
        }
    }
}


// Interface
void flex_conv_forward_kernel_cpu(
    at::Tensor features,
    at::Tensor theta,
    at::Tensor bias,
    at::Tensor neighborhood,
    at::Tensor positions,
    at::Tensor output)
{
    output.zero_();

    AT_DISPATCH_FLOATING_TYPES(
        features.type(), "flex_conv_forward_kernel_cpu", ([&] {
            flex_conv_forward_kernel_cpu_impl<scalar_t>(
                features.accessor<scalar_t, 3>(),
                theta.accessor<scalar_t, 3>(),
                bias.accessor<scalar_t, 2>(),
                neighborhood.accessor<int, 3>(),
                positions.accessor<scalar_t, 3>(),
                output.accessor<scalar_t, 3>());
        }));
}

void flex_conv_backward_kernel_cpu(
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
    grad_features.zero_();
    grad_theta.zero_();
    grad_bias.zero_();

    AT_DISPATCH_FLOATING_TYPES(
        features.type(), "flex_conv_backward_kernel_cpu", ([&] {
            flex_conv_backward_kernel_cpu_impl<scalar_t>(
                features.accessor<scalar_t, 3>(),
                theta.accessor<scalar_t, 3>(),
                bias.accessor<scalar_t, 2>(),
                neighborhood.accessor<int, 3>(),
                positions.accessor<scalar_t, 3>(),
                topdiff.accessor<scalar_t, 3>(),
                grad_features.accessor<scalar_t, 3>(),
                grad_theta.accessor<scalar_t, 3>(),
                grad_bias.accessor<scalar_t, 2>());
        }));
}
