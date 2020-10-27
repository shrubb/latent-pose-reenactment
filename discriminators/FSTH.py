import torch
from torch import nn
from torch.nn.utils import spectral_norm

import utils.radam
torch.optim.RAdam = utils.radam.RAdam

from generators.common import blocks
import math

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--dis_padding', type=str, default='zero', help='zero|reflection')
        parser.add('--dis_num_blocks', type=int, default=7)
        parser.add('--lr_dis', type=float, default=2e-4)

    @staticmethod
    def get_net(args):
        net = Discriminator(args.dis_padding, args.in_channels, args.out_channels, args.num_channels,
                            args.max_num_channels, args.embed_channels, args.dis_num_blocks, args.image_size,
                            args.num_labels).to(args.device)
        return net

    @staticmethod
    def get_optimizer(discriminator, args):
        Optimizer = torch.optim.__dict__[args.optimizer]
        return Optimizer(discriminator.parameters(), lr=args.lr_dis, betas=(args.beta1, 0.999), eps=1e-5)


class Discriminator(nn.Module):
    def __init__(self, padding, in_channels, out_channels, num_channels, max_num_channels, embed_channels,
                 dis_num_blocks, image_size,
                 num_labels):
        super().__init__()

        def get_down_block(in_channels, out_channels, padding):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=True,
                                   norm_layer='none')

        def get_res_block(in_channels, out_channels, padding):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=False,
                                   norm_layer='none')

        if padding == 'zero':
            padding = nn.ZeroPad2d
        elif padding == 'reflection':
            padding = nn.ReflectionPad2d

        self.out_channels = embed_channels
        in_channels = (in_channels + out_channels)

        self.down_block = nn.Sequential(
            padding(1),
            spectral_norm(
                nn.Conv2d(in_channels, num_channels, 3, 1, 0),
                eps=1e-4),
            nn.ReLU(),
            padding(1),
            spectral_norm(
                nn.Conv2d(num_channels, num_channels, 3, 1, 0),
                eps=1e-4),
            nn.AvgPool2d(2))
        self.skip = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels, num_channels, 1),
                eps=1e-4),
            nn.AvgPool2d(2))

        self.blocks = nn.ModuleList()
        num_down_blocks = min(int(math.log(image_size, 2)) - 2, dis_num_blocks)
        in_channels = num_channels
        for i in range(1, num_down_blocks):
            out_channels = min(in_channels * 2, max_num_channels)
            if i == dis_num_blocks - 1: out_channels = self.out_channels
            self.blocks.append(get_down_block(in_channels, out_channels, padding))
            in_channels = out_channels
        for i in range(num_down_blocks, dis_num_blocks):
            if i == dis_num_blocks - 1: out_channels = self.out_channels
            self.blocks.append(get_res_block(in_channels, out_channels, padding))

        self.linear = spectral_norm(nn.Linear(self.out_channels, 1), eps=1e-4)

        # Embeddings for identities
        embed = nn.Embedding(num_labels, self.out_channels)
        embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = spectral_norm(embed, eps=1e-4)

        self.finetuning = False

    def pass_inputs(self, input, embed=None):
        scores = []
        feats = []

        out = self.down_block(input)
        out = out + self.skip(input)
        feats.append(out)
        for block in self.blocks:
            out = block(out)
            feats.append(out)
        out = torch.relu(out)
        out = out.view(out.shape[0], self.out_channels, -1).sum(2)
        out_linear = self.linear(out)[:, 0]

        if embed is not None:
            scores.append((out * embed).sum(1) + out_linear)
        else:
            scores.append(out_linear)
        return scores[0], feats

    def enable_finetuning(self, data_dict=None):
        """
            Make the necessary adjustments to discriminator architecture to allow fine-tuning.
            For `vanilla` discriminator, replace embedding matrix W (`self.embed`) with one
            vector `data_dict['embeds']`.

            data_dict:
                dict
                Required contents depend on the specific discriminator. For `vanilla` discriminator,
                it is `'embeds'` (1 x `args.embed_channels`).
        """
        some_parameter = next(iter(self.parameters())) # to know target device and dtype

        if data_dict is None:
            data_dict = {
                'embeds': torch.rand(1, self.out_channels).to(some_parameter)
            }

        with torch.no_grad():
            if self.finetuning:
                self.embed.weight_orig.copy_(data_dict['embeds'])
            else:
                new_embedding_matrix = nn.Embedding(1, self.out_channels).to(some_parameter)
                new_embedding_matrix.weight.copy_(data_dict['embeds'])
                self.embed = spectral_norm(new_embedding_matrix)
                
                self.finetuning = True

    def forward(self, data_dict):
        fake_rgbs = data_dict['fake_rgbs']
        target_rgbs = data_dict['target_rgbs']
        dec_stickmen = data_dict['dec_stickmen']
        label = data_dict['label']

        if len(fake_rgbs.shape) > 4:
            fake_rgbs = fake_rgbs[:, 0]
        if len(target_rgbs.shape) > 4:
            target_rgbs = target_rgbs[:, 0]
        if len(dec_stickmen.shape) > 4:
            dec_stickmen = dec_stickmen[:, 0]

        b, c_in, h, w = dec_stickmen.shape

        embed = None
        if hasattr(self, 'embed'):
            embed = self.embed(label)

        disc_common = dec_stickmen

        fake_in = torch.cat([disc_common, fake_rgbs], dim=2).view(b, -1, h, w)
        fake_score_G, fake_features = self.pass_inputs(fake_in, embed)
        fake_score_D, _ = self.pass_inputs(fake_in.detach(), embed.detach())

        real_in = torch.cat([disc_common, target_rgbs], dim=2).view(b, -1, h, w)
        real_score, real_features = self.pass_inputs(real_in, embed)

        data_dict['fake_features'] = fake_features
        data_dict['real_features'] = real_features
        data_dict['real_embedding'] = embed
        data_dict['fake_score_G'] = fake_score_G
        data_dict['fake_score_D'] = fake_score_D
        data_dict['real_score'] = real_score
