import torch
from torch import nn
import torch.distributed as dist

from ._mst import mst
import numpy as np

# modified from https://github.com/Megvii-BaseDetection/TreeFilter-Torch, thanks for sharing the nice implementation.
class MinimumSpanningTree(nn.Module):
    def __init__(self, distance_func):
        super(MinimumSpanningTree, self).__init__()
        self.distance_func = distance_func

    @staticmethod
    def _build_matrix_index(fm):
        batch, height, width = (fm.shape[0], *fm.shape[2:])
        row = torch.arange(width, dtype=torch.int32, device=fm.device).unsqueeze(0)
        col = torch.arange(height, dtype=torch.int32, device=fm.device).unsqueeze(1)
        raw_index = row + col * width
        row_index = torch.stack([raw_index[:-1, :], raw_index[1:, :]], 2)
        col_index = torch.stack([raw_index[:, :-1], raw_index[:, 1:]], 2)
        index = torch.cat([row_index.reshape(1, -1, 2), col_index.reshape(1, -1, 2)], 1)
        index = index.expand(batch, -1, -1)
        return index

    def _build_feature_weight(self, fm):
        batch = fm.shape[0]
        weight_row = self.distance_func(fm[:, :, :-1, :], fm[:, :, 1:, :])
        weight_col = self.distance_func(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])
        weight = torch.cat([weight_row, weight_col], dim=1) + 1
        return weight

    def _build_label_weight(self, fm):
        batch = fm.shape[0]
        weight_row = self.distance_func(fm[:, :, :-1, :], fm[:, :, 1:, :])
        weight_col = self.distance_func(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])
        diff_weight = torch.cat([weight_row, weight_col], dim=1)

        weight_row = (fm[:, :, :-1, :] + fm[:, :, 1:, :]).sum(1)
        weight_col = (fm[:, :, :, :-1] + fm[:, :, :, 1:]).sum(1)
        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])
        labeled_weight = torch.cat([weight_row, weight_col], dim=1)

        weight = diff_weight * labeled_weight
        return weight

    def forward(self, guide_in):
        with torch.no_grad():
            index = self._build_matrix_index(guide_in)
            weight = self._build_feature_weight(guide_in)
            tree = mst(index, weight, guide_in.shape[2] * guide_in.shape[3])
        return tree

