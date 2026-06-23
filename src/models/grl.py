"""
Gradient Reversal Layer (GRL).

During the forward pass it is the identity. During the backward pass it
multiplies the gradient by -lambda. Placing it between the shared encoder and
the domain discriminator makes the encoder learn features that *maximise* the
discriminator's loss -> domain-invariant features. (Ganin & Lempitsky, 2015.)
"""

import numpy as np
import torch
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd: float = 1.0):
    return _GradReverse.apply(x, lambd)


def grl_lambda(step: int, total_steps: int, gamma: float = 10.0) -> float:
    """Standard DANN schedule: lambda ramps 0 -> 1 over training.

    p = step / total_steps;  lambda = 2/(1+exp(-gamma*p)) - 1
    """
    p = float(step) / max(1, total_steps)
    return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0
