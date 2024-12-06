from .losses import (GANLoss, MSELoss, g_path_regularize, compute_gradient_penalty,
                     r1_penalty)

__all__ = [
    'MSELoss', 'GANLoss', 'compute_gradient_penalty', 'r1_penalty', 'g_path_regularize'
]
