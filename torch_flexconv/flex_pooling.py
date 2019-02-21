import torch
import torch.nn as nn
import torch.autograd as autograd
from ._cuda_ext import flex_pool_forward, flex_pool_backward


class FlexMaxPoolFunction(autograd.Function):
    @staticmethod
    def forward(ctx, features, neighborhood):
        output, argmax = flex_pool_forward(features, neighborhood)

        ctx.save_for_backward(features, neighborhood, argmax)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, neighborhood, argmax = ctx.saved_variables

        grad_features = flex_pool_backward(
                features, neighborhood, grad_output, argmax
        )

        return grad_features, None


flex_maxpool = FlexMaxPoolFunction.apply


class FlexMaxPool(nn.Module):
    def forward(self, features, neighborhood):
        out = flex_maxpool(features, neighborhood)
        return out
