import torch


def init_softmax_(m, p):
    p = torch.tensor(p, dtype=torch.float32)
    z = torch.log(p / p[0])
    m.bias.data.copy_(z)
    return m