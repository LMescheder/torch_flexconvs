#ifndef USER_OPS_KERNELS_FLEX_POOL_OP_H_
#define USER_OPS_KERNELS_FLEX_POOL_OP_H_

#include "ATen/ATen.h"

void flex_pool_forward_kernel_cpu(
    at::Tensor features,
    at::Tensor neighborhood,
    at::Tensor output,
    at::Tensor argmax);

void flex_pool_backward_kernel_cpu(
    at::Tensor features,
    at::Tensor neighborhood,
    at::Tensor topdiff,
    at::Tensor argmax,
    at::Tensor grad_features);

void flex_pool_forward_kernel_cuda(
    at::Tensor features,
    at::Tensor neighborhood,
    at::Tensor output,
    at::Tensor argmax);

void flex_pool_backward_kernel_cuda(
    at::Tensor features,
    at::Tensor neighborhood,
    at::Tensor topdiff,
    at::Tensor argmax,
    at::Tensor grad_features);

#endif // USER_OPS_KERNELS_FLEX_POOL_OP_H_
