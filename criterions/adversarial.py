import torch
from torch import nn

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--gan_type', type=str, default='gan', help='gan|rgan|ragan')

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.gan_type)
        return criterion.to(args.device)


class Criterion(nn.Module):
    def __init__(self, gan_type):
        super().__init__()
        self.gan_type = gan_type

    def get_dis_preds(self, real_score, fake_score):
        if self.gan_type == 'gan':
            real_pred = real_score
            fake_pred = fake_score
        elif self.gan_type == 'rgan':
            real_pred = real_score - fake_score
            fake_pred = fake_score - real_score
        elif self.gan_type == 'ragan':
            real_pred = real_score - fake_score.mean()
            fake_pred = fake_score - real_score.mean()
        else:
            raise Exception('Incorrect `gan_type` argument')
        return real_pred, fake_pred

    def forward(self, data_dict):
        fake_score_G = data_dict['fake_score_G']
        fake_score_D = data_dict['fake_score_D']
        real_score = data_dict['real_score']

        real_pred, fake_pred_D = self.get_dis_preds(real_score, fake_score_D)
        _, fake_pred_G = self.get_dis_preds(real_score, fake_score_G)

        loss_D = torch.relu(1. - real_pred).mean() + torch.relu(1. + fake_pred_D).mean()  # TODO: check correctness

        if self.gan_type == 'gan':
            loss_G = -fake_pred_G.mean()
        elif 'r' in self.gan_type:
            loss_G = torch.relu(1. + real_pred).mean() + torch.relu(1. - fake_pred_G).mean()
        else:
            raise Exception('Incorrect `gan_type` argument')

        loss_G_dict = {}
        loss_G_dict['adversarial_G'] = loss_G

        loss_D_dict = {}
        loss_D_dict['adversarial_D'] = loss_D

        return loss_G_dict, loss_D_dict
