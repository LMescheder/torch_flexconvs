#include <torch/extension.h>
#include <vector>
#include "flex_conv.h"

torch::Tensor flex_conv_forward(
    torch::Tensor features,
    torch::Tensor theta,
    torch::Tensor bias,
    torch::Tensor neighborhood,
    torch::Tensor positions)
{
    // Determine dtype and device
    auto dtype = features.dtype();
    auto device = features.device();
    auto options = torch::dtype(dtype).device(device);

    // TODO: checks

    // Create output Tensor
    const int B = neighborhood.size(0);
    const int N = neighborhood.size(2);
    const int Dout = theta.size(2);
    auto output = torch::zeros({B, Dout, N}, options);

    // Run kernel
    if (device.is_cuda())
    {
        // flex_conv_forward_kernel_cuda(
        //     features, theta, bias, neighborhood, positions,
        //     output);
    }
    else
    {
        flex_conv_forward_kernel_cpu(
            features, theta, bias, neighborhood, positions,
            output);
    }

    return output;
}

std::vector<torch::Tensor> flex_conv_backward(
    torch::Tensor features,
    torch::Tensor theta,
    torch::Tensor bias,
    torch::Tensor neighborhood,
    torch::Tensor positions,
    torch::Tensor topdiff)
{
    // Determine dtype and device
    auto dtype = features.dtype();
    auto device = features.device();

    // TODO: checks

    // Create output tensors
    auto grad_features = torch::zeros_like(features);
    auto grad_theta = torch::zeros_like(theta);
    auto grad_bias = torch::zeros_like(bias);

    if (device.is_cuda())
    {
        flex_conv_backward_kernel_cuda(
            features, theta, bias,
            neighborhood, positions, topdiff,
            grad_features, grad_theta, grad_bias);
    }
    else
    {
        flex_conv_backward_kernel_cpu(
            features, theta, bias,
            neighborhood, positions, topdiff,
            grad_features, grad_theta, grad_bias);
    }

    return {grad_features, grad_theta, grad_bias};
}

// Interface
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flex_conv_forward", &flex_conv_forward, "FlexConv forward");
    m.def("flex_conv_backward", &flex_conv_backward, "FlexConv backward");
}