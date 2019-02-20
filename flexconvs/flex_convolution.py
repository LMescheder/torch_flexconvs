import torch
import torch.nn as nn
import torch.autograd as autograd
from _cuda_ext import flex_conv_forward, flex_conv_backward


class FlexConvolutionFunction(autograd.Function):
    @staticmethod
    def forward(ctx, features, weight_theta, weight_bias,
                neighborhood, positions, bias=None):
        output = flex_conv_forward(
            features, weight_theta, weight_bias, neighborhood, positions)

        if bias is not None:
            output = output + bias

        ctx.save_for_backward(
            features, weight_theta, weight_bias, neighborhood, positions)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, weight_theta, weight_bias, neighborhood, positions = \
            ctx.saved_variables

        grad_features, grad_theta, grad_bias = flex_conv_backward(
            features, grad_output, weight_bias,
            neighborhood, positions, grad_output
        )

        if weight_bias is not None:
            grad_weight_bias = grad_output
        else:
            grad_weight_bias = None

        gradients = (
            grad_features, grad_theta, grad_weight_bias, grad_output,
            None, None
        )

        return gradients


class FlexConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        # Parameters
        self.weight_theta = nn.Parameter(torch.zeros(3, in_channels, out_channels)) 
        self.weight_bias = nn.Parameter(torch.zeros(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_theta)
        nn.init.kaiming_uniform_(self.weight_bias)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, features, neighborhood, positions):
        out = FlexConvolutionFunction.apply(
            features, self.weight_theta, self.weight_bias,
            neighborhood, positions, self.bias
        )
        return out