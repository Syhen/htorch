# -*- coding: utf-8 -*-
"""
Author: @heyao

Created On: 2019/6/4 下午3:56
"""
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, epsilon=1e-10):
        """Attention layer.
        Compute weighted output of input tensor.
        :param feature_dim: int. input feature dim.
        :param step_dim: int. time step dim.
        :param bias: bool. weather use bias.
        :param epsilon: float.
        """
        super(Attention, self).__init__()

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.epsilon = epsilon

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + self.epsilon

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)
