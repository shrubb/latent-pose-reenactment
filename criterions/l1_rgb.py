from torch import nn

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--l1_weight', type=float, default=30.0)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.l1_weight)
        return criterion.to(args.device)

class Criterion(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, inputs):
        fake_rgb = inputs['fake_rgbs']
        real_rgb = inputs['target_rgbs']

        loss_G_dict = {}
        loss_G_dict['l1_rgb'] = self.weight * nn.functional.l1_loss(fake_rgb, real_rgb[:, 0])

        return loss_G_dict
