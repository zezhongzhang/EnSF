import numpy as np
import torch

# the lorenz drift
def lorenz96_drift(x, t):
    return (torch.roll(x, -1) - torch.roll(x, 2))*torch.roll(x, 1) - x + 8

def lorenz96_drift_np(x, t):
    return (np.roll(x, -1) - np.roll(x, 2))*np.roll(x, 1) - x + 8

def rk4(xt, fn, t, dt):
    k1 = fn(xt, t)
    k2 = fn(xt + dt / 2 * k1, t + dt / 2)
    k3 = fn(xt + dt / 2 * k2, t + dt / 2)
    k4 = fn(xt + dt * k3, t + dt)
    return xt + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def rmse_fn(x,y):
    return torch.sqrt(torch.mean( (x-y)**2)).item()

