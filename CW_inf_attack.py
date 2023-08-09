import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import torch.utils.data as data
import copy
import numpy as np
import torch.optim as optim


class CW_linf():
    def __init__(self, model, epsilon=8/255, alpha=2/255, attack_iters=50, restarts=1):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.attack_iters = attack_iters
        self.restarts = restarts

    def CW_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()

        loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
        return loss_value.mean()

    def __call__(self, X, y):
        max_loss = torch.zeros(y.shape[0]).cuda()
        max_delta = torch.zeros_like(X).cuda()
        # y_true = np.eye(10)[y.cuda().data.cpu().numpy()]
        # y_true = torch.from_numpy(y_true).cuda()
        upper_limit = 1.
        lower_limit = 0.
        for zz in range(self.restarts):
            delta = torch.zeros_like(X).cuda()
            delta.uniform_(-self.epsilon, self.epsilon)
            delta.data = torch.clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True
            for _ in range(self.attack_iters):
                output = self.model(X + delta)

                index = torch.where(output.max(1)[1] == y)
                if len(index[0]) == 0:
                    break
                loss = self.CW_loss(output, y)
                loss.backward()
                grad = delta.grad.detach()
                d = delta[index[0], :, :, :]
                g = grad[index[0], :, :, :]
                d = torch.clamp(d + self.alpha * torch.sign(g), -self.epsilon, self.epsilon)
                d = torch.clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
                delta.data[index[0], :, :, :] = d
                delta.grad.zero_()
            all_loss = F.cross_entropy(self.model(X + delta), y, reduction='none').detach()
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)
        return max_delta + X