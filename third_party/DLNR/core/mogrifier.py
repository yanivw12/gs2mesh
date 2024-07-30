import torch
from torch import nn


def weight(dim, factorize_k=None, kernel_size=3):
    if factorize_k is None:
        return nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2)

    return nn.Sequential(
        nn.Conv2d(dim, factorize_k, kernel_size, padding=kernel_size // 2),
        nn.Conv2d(factorize_k, dim, kernel_size, padding=kernel_size // 2)
    )


class Mogrifier(nn.Module):
    def __init__(self, dim, iters=5, factorize_k=None, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.iters = iters

        self.Q = weight(dim, factorize_k, kernel_size)
        self.R = weight(dim, factorize_k, kernel_size) if iters > 1 else None

    def forward(self, x, h):
        shape = x.shape
        print('x.shape')
        print(x.shape)
        *_, height, width = shape
        print('height')
        print(height)
        print('width')
        print(width)
        # assert dim == self.dim, f'mogrifier accepts a dimension of {self.dim}'

        # x, h = map(lambda t: t.reshape(-1, dim), (x, h))

        for ind in range(self.iters):
            if (ind % 2) == 0:
                x = 2 * self.Q(h).sigmoid() * x
            else:
                h = 2 * self.R(x).sigmoid() * h

        # x, h = map(lambda t: t.reshape(*shape), (x, h))
        return x, h


# m = Mogrifier(
#     dim=128,
#     iters=5,  # number of iterations, defaults to 5 as paper recommended for LSTM
#     factorize_k=16,  # factorize weight matrices into (dim x k) and (k x dim), if specified
#     kernel_size=3
# )
#
# x = torch.randn(8, 128, 384, 736)
# h = torch.randn(8, 128, 384, 736)
#
# x_out, h_out = m(x, h)  # (1, 16, 512), (1, 16, 512)
# print(x_out.shape)
# print(h_out.shape)
