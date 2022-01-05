import torch


def wasserstein_loss(y_true, y_pred):
    return -torch.mean(y_true * y_pred)
