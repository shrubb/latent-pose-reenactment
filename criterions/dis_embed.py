import torch
from torch import nn
import torch.nn.functional as F

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--dis_embed_weight', type=float, default=1e-2)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.dis_embed_weight)
        return criterion.to(args.device)


class Criterion(nn.Module):
    def __init__(self, dis_embed_weight):
        super().__init__()
        self.dis_embed_crit = lambda input, target: F.l1_loss(input, target.detach()) * dis_embed_weight

    def forward(self, data_dict):
        fake_embed = data_dict['embeds_elemwise']
        real_embed = data_dict['real_embedding']

        if len(fake_embed.shape) > 2:
            fake_embed = fake_embed[:, 0]

        if len(real_embed.shape) > 2:
            real_embed = real_embed[:, 0]

        loss_G_dict = {}
        loss_G_dict['embedding_matching'] = self.dis_embed_crit(fake_embed, real_embed)

        return loss_G_dict
