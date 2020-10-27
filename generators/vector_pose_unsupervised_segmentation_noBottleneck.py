import torch
from torch import nn
from torch.nn.utils import spectral_norm
from generators.common import blocks

import math

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--gen_constant_input_size', type=int, default=4)
        parser.add('--gen_num_residual_blocks', type=int, default=2)

        parser.add('--gen_padding', type=str, default='zero', help='zero|reflection')
        parser.add('--norm_layer', type=str, default='in')

    @staticmethod
    def get_net(args):
        # backward compatibility
        if 'gen_constant_input_size' not in args:
            args.gen_constant_input_size = 4

        net = Generator(
            args.gen_padding, args.in_channels, args.out_channels+1,
            args.num_channels, args.max_num_channels, args.embed_channels, args.pose_embedding_size,
            args.norm_layer, args.gen_constant_input_size, args.gen_num_residual_blocks,
            args.image_size)
        return net.to(args.device)


class Constant(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.constant = nn.Parameter(torch.ones(1, *shape))

    def forward(self, batch_size):
        return self.constant.expand((batch_size,) + self.constant.shape[1:])


class Generator(nn.Module):
    def __init__(self, padding, in_channels, out_channels, num_channels, max_num_channels, identity_embedding_size,
        pose_embedding_size, norm_layer, gen_constant_input_size, gen_num_residual_blocks, output_image_size):
        super().__init__()

        def get_res_block(in_channels, out_channels, padding, norm_layer):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=False,
                                   norm_layer=norm_layer)

        def get_up_block(in_channels, out_channels, padding, norm_layer):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=True, downsample=False,
                                   norm_layer=norm_layer)

        if padding == 'zero':
            padding = nn.ZeroPad2d
        elif padding == 'reflection':
            padding = nn.ReflectionPad2d
        else:
            raise Exception('Incorrect `padding` argument, required `zero` or `reflection`')

        assert math.log2(output_image_size / gen_constant_input_size).is_integer(), \
            "`gen_constant_input_size` must be `image_size` divided by a power of 2"
        num_upsample_blocks = int(math.log2(output_image_size / gen_constant_input_size))
        out_channels_block_nonclamped = num_channels * (2 ** num_upsample_blocks)
        out_channels_block = min(out_channels_block_nonclamped, max_num_channels)

        self.constant = Constant(out_channels_block, gen_constant_input_size, gen_constant_input_size)
        current_image_size = gen_constant_input_size

        # Decoder
        layers = []
        for i in range(gen_num_residual_blocks):
            layers.append(get_res_block(out_channels_block, out_channels_block, padding, 'ada' + norm_layer))
        
        for _ in range(num_upsample_blocks):
            in_channels_block = out_channels_block
            out_channels_block_nonclamped //= 2
            out_channels_block = min(out_channels_block_nonclamped, max_num_channels)
            layers.append(get_up_block(in_channels_block, out_channels_block, padding, 'ada' + norm_layer))

        layers.extend([
            blocks.AdaptiveNorm2d(out_channels_block, norm_layer),
            nn.ReLU(True),
            # padding(1),
            spectral_norm(
                nn.Conv2d(out_channels_block, out_channels, 3, 1, 1),
                eps=1e-4),
            nn.Tanh()
        ])
        self.decoder_blocks = nn.Sequential(*layers)

        self.adains = [module for module in self.modules() if module.__class__.__name__ == 'AdaptiveNorm2d']

        self.identity_embedding_size = identity_embedding_size
        self.pose_embedding_size = pose_embedding_size

        joint_embedding_size = identity_embedding_size + pose_embedding_size
        self.affine_params_projector = nn.Sequential(
            spectral_norm(nn.Linear(joint_embedding_size, max(joint_embedding_size, 512))),
            nn.ReLU(True),
            spectral_norm(nn.Linear(max(joint_embedding_size, 512), self.get_num_affine_params()))
        )

        self.finetuning = False

    def get_num_affine_params(self):
        return sum(2*module.num_features for module in self.adains)

    def assign_affine_params(self, affine_params):
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveNorm2d":
                new_bias = affine_params[:, :m.num_features]
                new_weight = affine_params[:, m.num_features:2 * m.num_features]

                if m.bias is None: # to keep m.bias being `nn.Parameter`
                    m.bias = new_bias.contiguous()
                else:
                    m.bias.copy_(new_bias)

                if m.weight is None: # to keep m.weight being `nn.Parameter`
                    m.weight = new_weight.contiguous()
                else:
                    m.weight.copy_(new_weight)

                if affine_params.size(1) > 2 * m.num_features:
                    affine_params = affine_params[:, 2 * m.num_features:]

    def assign_embeddings(self, data_dict):
        if self.finetuning:
            identity_embedding = self.identity_embedding.expand(len(data_dict['pose_embedding']), -1)
        else:
            identity_embedding = data_dict['embeds']
            
        pose_embedding = data_dict['pose_embedding']
        joint_embedding = torch.cat((identity_embedding, pose_embedding), dim=1)

        affine_params = self.affine_params_projector(joint_embedding)
        self.assign_affine_params(affine_params)

    def enable_finetuning(self, data_dict=None):
        """
            Make the necessary adjustments to generator architecture to allow fine-tuning.
            For `vanilla` generator, initialize AdaIN parameters from `data_dict['embeds']`
            and flag them as trainable parameters.
            Will require re-initializing optimizer, but only after the first call.

            data_dict:
                dict
                Required contents depend on the specific generator. For `vanilla` generator,
                it is `'embeds'` (1 x `args.embed_channels`).
                If `None`, the module's new parameters will be initialized randomly.
        """
        if data_dict is None:
            some_parameter = next(iter(self.parameters())) # to know target device and dtype
            identity_embedding = torch.rand(1, self.identity_embedding_size).to(some_parameter)
        else:
            identity_embedding = data_dict['embeds']

        if self.finetuning:
            with torch.no_grad():
                self.identity_embedding.copy_(identity_embedding)
        else:
            self.identity_embedding = nn.Parameter(identity_embedding)
            self.finetuning = True

    def forward(self, data_dict):
        self.assign_embeddings(data_dict)

        batch_size = len(data_dict['pose_embedding'])
        outputs = self.decoder_blocks(self.constant(batch_size))
        rgb, segmentation = outputs[:, :-1], outputs[:, -1:]

        # Move tanh's output from (-1; 1) to (-0.25; 1.25)
        rgb = rgb * 0.75
        rgb += 0.5

        # Same, but to (0; 1)
        segmentation = segmentation * 0.5
        segmentation += 0.5

        data_dict['fake_rgbs'] = rgb * segmentation
        data_dict['fake_segm'] = segmentation
