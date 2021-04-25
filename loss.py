import torch

def sqrt_loss(x, label):
    loss = x - label
    loss = 0.5 * loss * loss
    loss = torch.mean(loss)
    return loss