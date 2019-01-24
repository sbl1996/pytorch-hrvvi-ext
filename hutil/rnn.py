import unicodedata

from bidict import bidict

import torch
from hutil.common import cuda


def pad_sequence(sequences, batch_first=False, padding_value=0, max_len=None, padding='pre'):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max_len or max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            if padding == 'pre':
                out_tensor[i, max_len-length:, ...] = tensor
            else:
                out_tensor[i, :length, ...] = tensor
        else:
            if padding == 'pre':
                out_tensor[max_len-length:, i, ...] = tensor
            else:
                out_tensor[:length, i, ...] = tensor

    return out_tensor


class Vocab:

    def __init__(self, tokens=[], name='default'):
        super().__init__()
        self.name = name
        self._bd = bidict({})
        self.add(tokens)

    def __getitem__(self, token):
        return self._bd[token]

    def __len__(self):
        return len(self._bd)

    def _add_token(self, token):
        if token not in self._bd:
            self._bd[token] = len(self._bd)

    def add(self, tokens):
        for token in tokens:
            self._add_token(token)

    def tokens(self):
        return self._bd.keys()

    def get_index(self, token):
        return self._bd[token]

    def get_token_from_index(self, index):
        return self._bd.inv[index]

    def vocab_size(self):
        return len(self._bd)

    def as_tensor(self, tokens, out=None):
        n = len(tokens)
        if out is None:
            out = cuda(torch.empty(n, dtype=torch.long))
        for i, token in enumerate(tokens):
            out[i] = self.__getitem__(token)
        return out


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
