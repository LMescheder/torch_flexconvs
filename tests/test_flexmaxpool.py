import torch
from torch.autograd import gradcheck
from scipy.spatial import cKDTree
from torch_flexconv import FlexMaxPool, flex_maxpool


def test_flexmaxpool():
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

    model = FlexMaxPool()
    out1 = model(net, idx_nn)


    net = net.cuda()
    idx_nn = idx_nn.cuda()
    model = model.cuda()

    out2 = model(net, idx_nn).cpu()
    print(((out1 - out2).abs()/out1.abs()).mean())


def test_flexmaxpool_grads():
    B = 16
    in_channels = 8
    n_points = 20

    p = torch.rand(B, 3, n_points)
    p_np = p.squeeze().numpy()
    idx_nn = []
    for i in range(B):
        idx = cKDTree(p_np[i].T).query(p_np[i].T, k=3)[1]
        idx = torch.IntTensor(idx.T)
        idx_nn.append(idx)
    idx_nn = torch.stack(idx_nn, dim=0)

    feat = torch.rand(B, in_channels, n_points)

    # idx_nn = idx_nn.cuda()
    # feat = feat.cuda()

    feat = feat.to(torch.float64)
    p = p.to(torch.float64)

    feat.requires_grad_()

    gradcheck(
        flex_maxpool,
        [feat, idx_nn])


test_flexmaxpool()
test_flexmaxpool_grads()
