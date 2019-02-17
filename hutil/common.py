import torch


def one_hot(tensor, C=None, dtype=torch.float):
    d = tensor.dim()
    C = C or tensor.max() + 1
    t = tensor.new_zeros(*tensor.size(), C, dtype=dtype)
    return t.scatter_(d, tensor.unsqueeze(d), 1)


CUDA = torch.cuda.is_available()


def cuda(t):
    return t.cuda() if CUDA else t
