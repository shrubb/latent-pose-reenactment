from torch import nn
from criterions.common.perceptual_loss import PerceptualLoss

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--perc_weight', type=float, default=1e-2)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.perc_weight, args.vgg_weights_dir)
        return criterion.to(args.device)

class Criterion(nn.Module):
    def __init__(self, perc_weight, vgg_weights_dir):
        super().__init__()

        self.perceptual_crit = PerceptualLoss(perc_weight, vgg_weights_dir).eval()

    def forward(self, data_dict):
        fake_rgb = data_dict['fake_rgbs']
        real_rgb = data_dict['target_rgbs']

        if len(fake_rgb.shape) > 4:
            fake_rgb = fake_rgb[:, 0]

        if len(real_rgb.shape) > 4:
            real_rgb = real_rgb[:, 0]

        loss_G_dict = {}
        loss_G_dict['VGG'] = self.perceptual_crit(fake_rgb, real_rgb)

        return loss_G_dict
