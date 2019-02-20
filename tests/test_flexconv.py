import torch
from scipy.spatial import cKDTree
from flex_convolution import FlexConvolution


def test_flexconv():
    p = torch.rand(1, 3, 1000)
    p_np = p.squeeze().numpy().T
    idx_nn = cKDTree(p_np).query(p_np, k=9)[1][:, 1:]
    idx_nn = torch.IntTensor(idx_nn.T).unsqueeze(0)

    net = torch.rand(1, 32, 1000)

    conv = FlexConvolution(32, 32, bias=False)

    out1 = conv(net, idx_nn, p)

    net = net.cuda()
    idx_nn = idx_nn.cuda()
    p = p.cuda()
    conv = conv.cuda()

    out2 = conv(net, idx_nn, p)

    print(out1)
    print(out2)
