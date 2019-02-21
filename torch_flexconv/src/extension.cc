#include <torch/torch.h>
#include <vector>
#include "flex_conv.h"
#include "flex_deconv.h"
#include "flex_pool.h"


//Flexconv
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
        flex_conv_forward_kernel_cuda(
            features, theta, bias, neighborhood, positions,
            output);
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


// Flexdeconv

torch::Tensor flex_deconv_forward(
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
        flex_deconv_forward_kernel_cuda(
            features, theta, bias, neighborhood, positions,
            output);
    }
    else
    {
        flex_deconv_forward_kernel_cpu(
            features, theta, bias, neighborhood, positions,
            output);
    }

    return output;
}

std::vector<torch::Tensor> flex_deconv_backward(
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
        flex_deconv_backward_kernel_cuda(
            features, theta, bias,
            neighborhood, positions, topdiff,
            grad_features, grad_theta, grad_bias);
    }
    else
    {
        flex_deconv_backward_kernel_cpu(
            features, theta, bias,
            neighborhood, positions, topdiff,
            grad_features, grad_theta, grad_bias);
    }

    return {grad_features, grad_theta, grad_bias};
}


// FlexPool
std::vector<torch::Tensor> flex_pool_forward(
    torch::Tensor features,
    torch::Tensor neighborhood)
{
    // Determine dtype and device
    auto dtype = features.dtype();
    auto dtype_i = neighborhood.dtype();

    auto device = features.device();
    auto options = torch::dtype(dtype).device(device);
    auto options_i = torch::dtype(dtype_i).device(device);

    // TODO: checks

    // Create output Tensor
    const int B = neighborhood.size(0);
    const int N = neighborhood.size(2);
    const int D = features.size(1);

    auto output = torch::zeros({B, D, N}, options);
    auto argmax = torch::zeros({B, D, N}, options_i);

    // Run kernel
    if (device.is_cuda())
    {
        flex_pool_forward_kernel_cuda(
            features, neighborhood, output, argmax);
    }
    else
    {
        flex_pool_forward_kernel_cpu(
            features, neighborhood, output, argmax);
    }

    return {output, argmax};
}

torch::Tensor flex_pool_backward(
    torch::Tensor features,
    torch::Tensor neighborhood,
    torch::Tensor topdiff,
    torch::Tensor argmax)
{
    // Determine dtype and device
    auto dtype = features.dtype();
    auto device = features.device();

    // TODO: checks

    // Create output tensors
    auto grad_features = torch::zeros_like(features);

    if (device.is_cuda())
    {
        flex_pool_backward_kernel_cuda(
            features, neighborhood, topdiff, argmax, grad_features);
    }
    else
    {
        flex_pool_backward_kernel_cpu(
            features, neighborhood, topdiff, argmax, grad_features);

    }

    return grad_features;
}

// Interface
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flex_conv_forward", &flex_conv_forward, "FlexConv forward");
    m.def("flex_conv_backward", &flex_conv_backward, "FlexConv backward");
    m.def("flex_deconv_forward", &flex_deconv_forward, "FlexDeconv forward");
    m.def("flex_deconv_backward", &flex_deconv_backward, "FlexDeconv backward");
    m.def("flex_pool_forward", &flex_pool_forward, "FlexPool forward");
    m.def("flex_pool_backward", &flex_pool_backward, "FlexPool backward");
}