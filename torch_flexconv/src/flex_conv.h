#ifndef USER_OPS_KERNELS_FLEX_CONV_OP_H_
#define USER_OPS_KERNELS_FLEX_CONV_OP_H_

#include "ATen/ATen.h"

void flex_conv_forward_kernel_cpu(
    at::Tensor features,
    at::Tensor theta,
    at::Tensor bias,
    at::Tensor neighborhood,
    at::Tensor positions,
    at::Tensor output);

void flex_conv_backward_kernel_cpu(
    at::Tensor features,
    at::Tensor theta,
    at::Tensor bias,
    at::Tensor neighborhood,
    at::Tensor positions,
    at::Tensor topdiff,
    at::Tensor grad_features,
    at::Tensor grad_theta,
    at::Tensor grad_bias);

void flex_conv_forward_kernel_cuda(
    at::Tensor features,
    at::Tensor theta,
    at::Tensor bias,
    at::Tensor neighborhood,
    at::Tensor positions,
    at::Tensor output);

void flex_conv_backward_kernel_cuda(
    at::Tensor features,
    at::Tensor theta,
    at::Tensor bias,
    at::Tensor neighborhood,
    at::Tensor positions,
    at::Tensor topdiff,
    at::Tensor grad_features,
    at::Tensor grad_theta,
    at::Tensor grad_bias);

#endif // USER_OPS_KERNELS_FLEX_CONV_OP_H_
