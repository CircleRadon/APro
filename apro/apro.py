import torch
import torch.nn as nn
from .gp_cuda.gp_process.GP import gp_process

class Global_APro(nn.Module):
    """
    implementation of global affinity propagation
    """
    def __init__(self):
        super(Global_APro, self).__init__()

    @staticmethod
    def get_true_index(data, index):
        with torch.no_grad():
            C = data.shape[1]
            index = index.unsqueeze(1).expand(-1, C, -1).long()
        data = torch.gather(data, 2, index)
        return data
    
    @staticmethod
    def norm2_distance(src, tgt):
        diff = src - tgt
        dis = (diff * diff).sum(dim=1)
        return dis

    def forward(self, feature_in, embed_in, tree, zeta_g=0.01):
        source_idx = tree[:, :, 0]
        target_idx = tree[:, :, 1]

        bs, c, h, w = embed_in.shape
        embed_in = embed_in.reshape(bs, c, -1)

        source_node = self.get_true_index(embed_in, source_idx)
        target_node = self.get_true_index(embed_in, target_idx)
        edge_weight = self.norm2_distance(source_node, target_node)

        feature_in = feature_in.reshape(bs, -1)

        feature_out = gp_process(tree, edge_weight, feature_in, zeta_g)

        feature_out = feature_out.reshape(bs, 1, h, w)

        return feature_out


class Local_APro(nn.Module):
    """
    implementation of local affinity propagation
    """
    def __init__(self, kernel_size=5, zeta_s=0.15, num_iter=20):
        super(Local_APro, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 == 1
        self.zeta_s = zeta_s
        self.num_iter = num_iter
        self.unfold = torch.nn.Unfold(self.kernel_size, stride=1, padding=self.kernel_size // 2)

    @torch.no_grad()
    def forward(self, img, feat, masked_box=None):
        img = img.float()
        B, H, W = feat.shape
        C = img.shape[1]

        img = img + 10  #differ from zeros padding
        unfold_img = self.unfold(img).reshape(B, C, self.kernel_size ** 2, H * W)
        aff = torch.exp(-(((unfold_img - img.reshape(B, C, 1, H * W)) ** 2) / (self.zeta_s ** 2)).sum(1))

        if masked_box is not None:
            masked_box = masked_box.reshape(B, H, W)
            feat = feat * masked_box
        else:
            masked_box = None

        for it in range(self.num_iter):
            feat = self.single_forward(feat, aff, masked_box)

        return feat

    def single_forward(self, x, aff, masked_box):
        
        B, H, W = x.shape
        unfold_x = self.unfold(x[:, None]).squeeze(1).reshape(B, self.kernel_size ** 2, H * W)

        propa = (unfold_x * aff).sum(1)
        sumz = aff.sum(1)
        propa = propa/(sumz+1e-10)

        if masked_box is not None:
            propa = propa * masked_box.reshape(B, H * W)

        return propa.reshape(B, H, W)
