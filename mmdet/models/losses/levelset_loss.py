import torch
import torch.nn as nn
from ..builder import LOSSES

@LOSSES.register_module()
class LevelsetLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(LevelsetLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, mask_logits, targets, pixel_num):
        region_levelset_term = region_levelset()
        # length_regu_term = length_regularization()

        region_levelset_loss = region_levelset_term(mask_logits, targets) / pixel_num
        # length_regu = 0.00001 * length_regu_term(mask_logits) / pixel_num

        loss_levelst = self.loss_weight * region_levelset_loss
        # loss_levelst = self.loss_weight * region_levelset_loss + length_regu

        return loss_levelst


class region_levelset(nn.Module):
    '''
    The mian of region leveset function.
    '''

    def __init__(self):
        super(region_levelset, self).__init__()

    def forward(self, mask_score, lst_target):
        '''
        mask_score: predcited mask scores        tensor:(N,2,W,H) 
        lst_target:  input target for levelset   tensor:(N,C,W,H) 
        '''
        
        mask_score_f = mask_score[:, 0, :, :].unsqueeze(1)
        mask_score_b = mask_score[:, 1, :, :].unsqueeze(1)
        interior_ = torch.sum(mask_score_f * lst_target, (2, 3)) / torch.sum(mask_score_f, (2, 3)).clamp(min=0.00001)
        exterior_ = torch.sum(mask_score_b * lst_target, (2, 3)) / torch.sum(mask_score_b, (2, 3)).clamp(min=0.00001)
        interior_region_level = torch.pow(lst_target - interior_.unsqueeze(-1).unsqueeze(-1), 2)
        exterior_region_level = torch.pow(lst_target - exterior_.unsqueeze(-1).unsqueeze(-1), 2)
        region_level_loss = interior_region_level*mask_score_f + exterior_region_level*mask_score_b
        level_set_loss = torch.sum(region_level_loss, (1, 2, 3))/lst_target.shape[1]
        
        return level_set_loss


class length_regularization(nn.Module):

    '''
    calcaulate the length by the gradient for regularization.
    '''

    def __init__(self):
        super(length_regularization, self).__init__()

    def forward(self, mask_score):
        gradient_H = torch.abs(mask_score[:, :, 1:, :] - mask_score[:, :, :-1, :])
        gradient_W = torch.abs(mask_score[:, :, :, 1:] - mask_score[:, :, :, :-1])
        curve_length = torch.sum(gradient_H, dim=(1,2,3)) + torch.sum(gradient_W, dim=(1,2,3))
        return curve_length



def LCM(imgs, pred_phis, box_targets):

    lcm = LocalConsistencyModule(num_iter=10, dilations=[2]).to(pred_phis.device)
    refine_phis = lcm(imgs, pred_phis)
    local_consist = (torch.abs(refine_phis - pred_phis) * box_targets).sum()
    local_regions = box_targets.sum().clamp(min=1)
    return local_consist / local_regions


class LocalConsistencyModule(nn.Module):
    """
    Local Consistency Module (LCM) for Level set phi prediction.
    """

    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = self.get_kernel()
        self.register_buffer('kernel', kernel)
        self.alpha = 0.3

    def get_kernel(self):

        kernel = torch.zeros(8, 1, 3, 3)
        kernel[0, 0, 0, 0] = 1
        kernel[1, 0, 0, 1] = 1
        kernel[2, 0, 0, 2] = 1
        kernel[3, 0, 1, 0] = 1
        kernel[4, 0, 1, 2] = 1
        kernel[5, 0, 2, 0] = 1
        kernel[6, 0, 2, 1] = 1
        kernel[7, 0, 2, 2] = 1

        return kernel

    def get_dilated_neighbors(self, x):
        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)
        return torch.cat(x_aff, dim=2)

    def forward(self, imgs, pred_phis):

        _imgs = self.get_dilated_neighbors(imgs)
        _imgs_rep = imgs.unsqueeze(2).repeat(1, 1, _imgs.shape[2], 1, 1)

        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=2, keepdim=True)

        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.alpha) ** 2
        aff = aff.mean(dim=1, keepdim=True)
        aff = F.softmax(aff, dim=2)

        for _ in range(self.num_iter):
            _pred_phis = self.get_dilated_neighbors(pred_phis)
            refine_phis = (_pred_phis * aff).sum(2)

        return refine_phis