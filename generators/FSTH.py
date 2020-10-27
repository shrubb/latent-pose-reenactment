import torch
from torch import nn
from torch.nn.utils import spectral_norm
from generators.common import blocks

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--gen_padding', type=str, default='zero', help='zero|reflection')
        parser.add('--gen_num_downsample_blocks', type=int, default=4)
        parser.add('--gen_num_residual_blocks', type=int, default=4)
        parser.add('--norm_layer', type=str, default='in')

    @staticmethod
    def get_net(args):
        net = Generator(
            args.gen_padding, args.in_channels, args.out_channels,
            args.num_channels, args.max_num_channels, args.embed_channels,
            args.norm_layer, args.gen_num_downsample_blocks, args.gen_num_residual_blocks)
        return net.to(args.device)


class Generator(nn.Module):
    def __init__(self, padding, in_channels, out_channels, num_channels, max_num_channels, embed_channels, norm_layer,
                 gen_num_downsample_blocks, gen_num_residual_blocks):
        super().__init__()

        def get_down_block(in_channels, out_channels, padding, norm_layer):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=True,
                                   norm_layer=norm_layer)

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

        in_channels_block = in_channels

        # Encoder of inputs
        self.down_block = nn.Sequential(
            padding(1),
            spectral_norm(
                nn.Conv2d(in_channels_block, num_channels, 3, 1, 0),
                eps=1e-4),
            nn.ReLU(),
            padding(1),
            spectral_norm(
                nn.Conv2d(num_channels, num_channels, 3, 1, 0),
                eps=1e-4),
            nn.AvgPool2d(2))
        self.skip = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels_block, num_channels, 1),
                eps=1e-4),
            nn.AvgPool2d(2))
        layers = []
        in_channels_block = num_channels
        for i in range(1, gen_num_downsample_blocks):
            out_channels_block = min(in_channels_block * 2, max_num_channels)
            layers.append(get_down_block(in_channels_block, out_channels_block, padding, norm_layer))
            in_channels_block = out_channels_block
        self.down_blocks = nn.Sequential(*layers)

        # Decoder
        layers = []
        for i in range(gen_num_residual_blocks):
            layers.append(get_res_block(out_channels_block, out_channels_block, padding, 'ada' + norm_layer))
        for i in range(gen_num_downsample_blocks - 1, -1, -1):
            in_channels_block = out_channels_block
            out_channels_block = min(int(num_channels * 2 ** i), max_num_channels)
            layers.append(get_up_block(in_channels_block, out_channels_block, padding, 'ada' + norm_layer))
        layers.extend([
            blocks.AdaptiveNorm2d(out_channels_block, norm_layer),
            nn.ReLU(True),
            padding(1),
            spectral_norm(
                nn.Conv2d(out_channels_block, out_channels, 3, 1, 0),
                eps=1e-4),
            nn.Tanh()])
        self.decoder_blocks = nn.Sequential(*layers)

        # self.project moved from embedder
        num_affine_params = self.get_num_affine_params()

        self.project = spectral_norm(
            nn.Linear(embed_channels, num_affine_params),
            eps=1e-4)

        self.finetuning = False

    def get_num_affine_params(self):
        num_affine_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveNorm2d":
                num_affine_params += 2 * m.num_features
        return num_affine_params

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
        embedding = data_dict['embeds']
        affine_params = self.project(embedding)
        self.assign_affine_params(affine_params)

    def make_affine_params_trainable(self):
        """
            Used prior to fine-tuning.

            Flag `.weight` and `.bias` of all `AdaptiveNorm2d` layers as trainable parameters.
            After calling this function, the said tensors will be returned by `.parameters()`.
            Their values are set to those present at the time of calling this function, i.e.
            `.assign_embeddings()` (or `.assign_affine_params`) must be called beforehand.
        """
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveNorm2d":
                for field in m.weight, m.bias:
                    assert torch.is_tensor(field) and field.numel() == m.num_features, \
                        "One of `AdaptiveNorm2d`'s parameters is of wrong size or is None. " \
                        "Did you forget to call `.assign_embeddings()` (or `.assign_affine_params()`)?"

                m.weight = nn.Parameter(m.weight)
                m.bias   = nn.Parameter(m.bias)
                m.delete_weight_on_forward = False

    def enable_finetuning(self, data_dict=None):
        """
            Make the necessary adjustments to generator architecture to allow fine-tuning.
            For `vanilla` generator, initialize AdaIN parameters from `data_dict['embeds']`
            and flag them as trainable parameters.
            Will require re-initializing optimizer, but only after the first call.

            data_dict:
                dict, optional
                Required contents depend on the specific generator. For `vanilla` generator,
                it is `'embeds'` (1 x `args.embed_channels`).
                If `None`, the module's new parameters will be initialized randomly.
        """
        was_training = self.training
        self.eval() # TODO use `args.set_eval_mode_in_test` instead of hard `True`

        if data_dict is None:
            some_parameter = next(iter(self.parameters())) # to know target device and dtype
            data_dict = {
                'embeds': torch.rand(1, self.project.in_features).to(some_parameter)
            }

        with torch.no_grad():
            self.assign_embeddings(data_dict)
        
        if not self.finetuning:
            self.make_affine_params_trainable()
            self.finetuning = True

        self.train(was_training)

    def forward(self, data_dict):
        if not self.finetuning: # made `True` in `.make_affine_params_trainable()` (e.g. for fine-tuning)
            self.assign_embeddings(data_dict)
        
        inputs = data_dict['dec_stickmen']
        if len(inputs.shape) > 4:
            inputs = inputs[:, 0]

        out = self.down_block(inputs)
        out = out + self.skip(inputs)
        out = self.down_blocks(out)
        # Decode
        outputs = self.decoder_blocks(out)

        data_dict['fake_rgbs'] = outputs
