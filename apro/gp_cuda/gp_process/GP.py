import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import GP_cuda as _C

class _GP_Process(Function):
    @staticmethod
    def forward(ctx, edges, edge_weight, prob, sigma):
        res = _C.gp_forward(edges, edge_weight, prob, sigma)
        return res

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return None, None, None, None

gp_process = _GP_Process.apply

