from torch import nn
import torch.nn.functional as F

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--fm_weight', type=float, default=10.0)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.fm_weight)
        return criterion.to(args.device)


class Criterion(nn.Module):
    def __init__(self, fm_weight):
        super().__init__()
        self.fm_crit = lambda inputs, targets: sum(
            [F.l1_loss(input, target.detach()) for input, target in zip(inputs, targets)]) / len(
            inputs) * fm_weight

    def forward(self, data_dict):
        fake_feats = data_dict['fake_features']
        real_feats = data_dict['real_features']

        loss_G_dict = {}
        loss_G_dict['feature_matching'] = self.fm_crit(fake_feats, real_feats)

        return loss_G_dict
