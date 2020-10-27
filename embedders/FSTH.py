import torch
from torch import nn
from torch.nn.utils import spectral_norm
from generators.common import blocks

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--embed_padding', type=str, default='zero', help='zero|reflection')
        parser.add('--embed_num_blocks', type=int, default=6)
        parser.add('--average_function', type=str, default='sum', help='sum|max')

    @staticmethod
    def get_net(args):
        net = Embedder(
            args.embed_padding, args.in_channels, args.out_channels,
            args.num_channels, args.max_num_channels, args.embed_channels,
            args.embed_num_blocks, args.average_function)
        return net.to(args.device)

class Embedder(nn.Module):
    def __init__(self, padding, in_channels, out_channels, num_channels, max_num_channels, embed_channels,
                 embed_num_blocks, average_function):
        super().__init__()

        def get_down_block(in_channels, out_channels, padding):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=True,
                                   norm_layer='none')

        if padding == 'zero':
            padding = nn.ZeroPad2d
        elif padding == 'reflection':
            padding = nn.ReflectionPad2d

        self.out_channels = embed_channels

        self.down_block = nn.Sequential(
            padding(1),
            spectral_norm(
                nn.Conv2d(in_channels + out_channels, num_channels, 3, 1, 0),
                eps=1e-4),
            nn.ReLU(),
            padding(1),
            spectral_norm(
                nn.Conv2d(num_channels, num_channels, 3, 1, 0),
                eps=1e-4),
            nn.AvgPool2d(2))
        self.skip = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels + out_channels, num_channels, 1),
                eps=1e-4),
            nn.AvgPool2d(2))

        layers = []
        in_channels = num_channels
        for i in range(1, embed_num_blocks - 1):
            out_channels = min(in_channels * 2, max_num_channels)
            layers.append(get_down_block(in_channels, out_channels, padding))
            in_channels = out_channels
        layers.append(get_down_block(out_channels, embed_channels, padding))
        self.down_blocks = nn.Sequential(*layers)

        self.average_function = average_function

        self.finetuning = False

    def enable_finetuning(self, data_dict=None):
        self.finetuning = True

    def get_identity_embedding(self, data_dict):
        enc_stickmen = data_dict['enc_stickmen']
        enc_rgbs = data_dict['enc_rgbs']

        inputs = torch.cat([enc_stickmen, enc_rgbs], 2)

        b, n, c, h, w = inputs.shape
        inputs = inputs.view(-1, c, h, w)
        out = self.down_block(inputs)
        out = out + self.skip(inputs)
        out = self.down_blocks(out)
        out = torch.relu(out)
        embeds_elemwise = out.view(b, n, self.out_channels, -1).sum(3)

        if self.average_function == 'sum':
            embeds = embeds_elemwise.mean(1)
        elif self.average_function == 'max':
            embeds = embeds_elemwise.max(1)[0]
        else:
            raise Exception('Incorrect `average_function` argument, expected `sum` or `max`')

        data_dict['embeds'] = embeds
        data_dict['embeds_elemwise'] = embeds_elemwise

    def get_pose_embedding(self, data_dict):
        pass

    def forward(self, data_dict):
        if not self.finetuning:
            self.get_identity_embedding(data_dict)
        self.get_pose_embedding(data_dict)
