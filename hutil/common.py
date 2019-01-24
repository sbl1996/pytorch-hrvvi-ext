import torch


def one_hot(tensor, C=None):
    d = tensor.dim()
    C = C or tensor.max() + 1
    t = tensor.new_zeros(*tensor.size(), C, dtype=torch.float)
    return t.scatter_(d, tensor.unsqueeze(d), 1)


CUDA = torch.cuda.is_available()


def cuda(t):
    return t.cuda() if CUDA else t
