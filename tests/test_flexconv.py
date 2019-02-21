import torch
from torch.autograd import gradcheck
from scipy.spatial import cKDTree
from flex_convolution import FlexConvolution, flex_convolution


def test_flexconv():
    B = 16
    p = torch.rand(B, 3, 1000)
    p_np = p.squeeze().numpy()
    idx_nn = []
    for i in range(B):
        idx = cKDTree(p_np[i].T).query(p_np[i].T, k=12)[1]
        idx = torch.IntTensor(idx.T)
        idx_nn.append(idx)
    idx_nn = torch.stack(idx_nn, dim=0)

    net = torch.rand(B, 32, 1000)
    grad_in = torch.rand(B, 32, 1000)

    conv = FlexConvolution(32, 32)

    conv.zero_grad()
    out1 = conv(net, idx_nn, p)
    out1.backward(grad_in)
    out1 = out1.cpu().detach()
    grad_theta1 = conv.weight_theta.grad.cpu().detach()
    grad_bias1 = conv.weight_bias.grad.cpu().detach()

    net = net.cuda()
    idx_nn = idx_nn.cuda()
    p = p.cuda()
    grad_in = grad_in.cuda()
    conv = conv.cuda()

    conv.zero_grad()
    out2 = conv(net, idx_nn, p)
    out2.backward(grad_in)
    out2 = out2.cpu().detach()
    grad_theta2 = conv.weight_theta.grad.cpu().detach()
    grad_bias2 = conv.weight_bias.grad.cpu().detach()

    print(((out1 - out2).abs()/out1.abs()).mean())
    print(((grad_theta1 - grad_theta2).abs()/grad_theta1.abs()).mean())
    print(((grad_bias1 - grad_bias2).abs()/grad_bias1.abs()).mean())


def test_flexconv_grads():
    B = 16
    in_channels = 8
    out_channels = 4
    n_points = 20

    p = torch.rand(B, 3, n_points)
    p_np = p.squeeze().numpy()
    idx_nn = []
    for i in range(B):
        idx = cKDTree(p_np[i].T).query(p_np[i].T, k=3)[1]
        idx = torch.IntTensor(idx.T)
        idx_nn.append(idx)
    idx_nn = torch.stack(idx_nn, dim=0)

    weight_theta = torch.randn(3, in_channels, out_channels)
    weight_bias = torch.randn(in_channels, out_channels)
    bias = torch.randn(out_channels)
    feat = torch.rand(B, in_channels, n_points)

    # weight_theta = weight_theta.cuda()
    # weight_bias = weight_bias.cuda()
    # bias = bias.cuda()
    # p = p.cuda()
    # idx_nn = idx_nn.cuda()
    # feat = feat.cuda()

    feat = feat.to(torch.float64)
    weight_theta = weight_theta.to(torch.float64)
    weight_bias = weight_bias.to(torch.float64)
    bias = bias.to(torch.float64)
    p = p.to(torch.float64)

    feat.requires_grad_()
    weight_theta.requires_grad_()
    weight_bias.requires_grad_()
    bias.requires_grad_()

    gradcheck(
        flex_convolution, 
        [feat, weight_theta, weight_bias, idx_nn, p, bias])
