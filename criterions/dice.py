import torch
from torch import nn

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--dice_weight', type=float, default=1)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.dice_weight)
        return criterion.to(args.device)


class Criterion(nn.Module):
    def __init__(self, dice_weight):
        super().__init__()
        self.dice_weight = dice_weight

    def forward(self, data_dict):
        fake_segm = data_dict['fake_segm']
        real_segm = data_dict['real_segm']

        if len(fake_segm.shape) > 4:
            fake_segm = fake_segm[:, 0]

        if len(real_segm.shape) > 4:
            real_segm = real_segm[:, 0]

        numer = (2*fake_segm*real_segm).sum()
        denom =  ((fake_segm**2).sum() + (real_segm**2).sum())

        dice = numer / denom
        loss = -torch.log(dice) * self.dice_weight

        loss_G_dict = {}
        loss_G_dict['segmentation_dice'] = loss

        return loss_G_dict
