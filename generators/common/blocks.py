import torch
from torch import nn
from torch.nn.utils import spectral_norm


class AdaptiveNorm2d(nn.Module):
    def __init__(self, num_features, norm_layer='in', eps=1e-4):
        super(AdaptiveNorm2d, self).__init__()
        self.num_features = num_features
        self.weight = self.bias = None
        if 'in' in norm_layer:
            self.norm_layer = nn.InstanceNorm2d(num_features, eps=eps, affine=False)
        elif 'bn' in norm_layer:
            self.norm_layer = SyncBatchNorm(num_features, momentum=1.0, eps=eps, affine=False)

        self.delete_weight_on_forward = True

    def forward(self, input):
        out = self.norm_layer(input)
        output = out * self.weight[:, :, None, None] + self.bias[:, :, None, None]

        # To save GPU memory
        if self.delete_weight_on_forward:
            self.weight = self.bias = None

        return output


class AdaptiveNorm2dTrainable(nn.Module):
    def __init__(self, num_features, norm_layer='in', eps=1e-4):
        super(AdaptiveNorm2dTrainable, self).__init__()
        self.num_features = num_features
        if 'in' in norm_layer:
            self.norm_layer = nn.InstanceNorm2d(num_features, eps=eps, affine=False)

    def forward(self, input):
        out = self.norm_layer(input)
        t = out.shape[0] // self.weight.shape[0]
        output = out * self.weight + self.bias
        return output

    def assign_params(self, weight, bias):
        self.weight = torch.nn.Parameter(weight.view(1, -1, 1, 1))
        self.bias = torch.nn.Parameter(bias.view(1, -1, 1, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding, upsample, downsample,
                 norm_layer, activation=nn.ReLU, gated=False):
        super(ResBlock, self).__init__()
        normalize = norm_layer != 'none'
        bias = not normalize

        # if norm_layer == 'bn':
        #     # norm0 = SyncBatchNorm(in_channels, momentum=1.0, eps=1e-4)
        #     # norm1 = SyncBatchNorm(out_channels, momentum=1.0, eps=1e-4)
        #     pass
        if norm_layer == 'in':
            norm0 = nn.InstanceNorm2d(in_channels, eps=1e-4, affine=True)
            norm1 = nn.InstanceNorm2d(out_channels, eps=1e-4, affine=True)
        elif 'ada' in norm_layer:
            norm0 = AdaptiveNorm2d(in_channels, norm_layer)
            norm1 = AdaptiveNorm2d(out_channels, norm_layer)
        elif 'tra' in norm_layer:
            norm0 = AdaptiveNorm2dTrainable(in_channels, norm_layer)
            norm1 = AdaptiveNorm2dTrainable(out_channels, norm_layer)
        elif normalize:
            raise Exception('ResBlock: Incorrect `norm_layer` parameter')

        layers = []
        if normalize:
            layers.append(norm0)
        layers.append(activation(inplace=True))
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        layers.extend([
            nn.Sequential() if padding is nn.ZeroPad2d else padding(1),
            spectral_norm(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1 if padding is nn.ZeroPad2d else 0, bias=bias),
                eps=1e-4)])
        if normalize:
            layers.append(norm1)
        layers.extend([
            activation(inplace=True),
            nn.Sequential() if padding is nn.ZeroPad2d else padding(1),
            spectral_norm(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1 if padding is nn.ZeroPad2d else 0, bias=bias),
                eps=1e-4)])
        if downsample:
            layers.append(nn.AvgPool2d(2))
        self.block = nn.Sequential(*layers)

        self.skip = None
        if in_channels != out_channels or upsample or downsample:
            layers = []
            if upsample:
                layers.append(nn.Upsample(scale_factor=2))
            layers.append(spectral_norm(
                nn.Conv2d(in_channels, out_channels, 1),
                eps=1e-4))
            if downsample:
                layers.append(nn.AvgPool2d(2))
            self.skip = nn.Sequential(*layers)

    def forward(self, input):
        out = self.block(input)
        if self.skip is not None:
            output = out + self.skip(input)
        else:
            output = out + input
        return output

class channelShuffle(nn.Module):
    def __init__(self,groups):
        super(channelShuffle, self).__init__()
        self.groups=groups
        
    def forward(self,x):
        batchsize, num_channels, height, width = x.data.size()

#         batchsize = x.shape[0]
#         num_channels = x.shape[1]
#         height = x.shape[2]
#         width = x.shape[3]

        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        # transpose
        # - contiguous() required if transpose() is used before view().
        #   See https://github.com/pytorch/pytorch/issues/764
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x
    

class shuffleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(shuffleConv, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.stride=stride
        self.padding=padding
        groups=4
        block=[]
        if (in_channels%groups==0) and (out_channels%groups==0):
            block.append(spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,padding=0, groups=groups),eps=1e-4))
            block.append(nn.ReLU6(inplace=True))
            block.append(channelShuffle(groups=groups))
            block.append(spectral_norm(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,kernel_size=3,padding=1, groups=groups),eps=1e-4))
            block.append(nn.ReLU6(inplace=True))
            block.append(spectral_norm(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,kernel_size=1,padding=0, groups=groups),eps=1e-4))
        else:
            block.append(spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,padding=1),eps=1e-4))
        self.block=nn.Sequential(*block)
            
    def forward(self,x):
        x=self.block(x)
        return x    
    
    
class ResBlockShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, padding, upsample, downsample,
                 norm_layer, activation=nn.ReLU, gated=False):
        super(ResBlockShuffle, self).__init__()
        normalize = norm_layer != 'none'
        bias = not normalize

        # if norm_layer == 'bn':
        #     # norm0 = SyncBatchNorm(in_channels, momentum=1.0, eps=1e-4)
        #     # norm1 = SyncBatchNorm(out_channels, momentum=1.0, eps=1e-4)
        #     pass
        if norm_layer == 'in':
            norm0 = nn.InstanceNorm2d(in_channels, eps=1e-4, affine=True)
            norm1 = nn.InstanceNorm2d(out_channels, eps=1e-4, affine=True)
        elif 'ada' in norm_layer:
            norm0 = AdaptiveNorm2d(in_channels, norm_layer)
            norm1 = AdaptiveNorm2d(out_channels, norm_layer)
        elif 'tra' in norm_layer:
            norm0 = AdaptiveNorm2dTrainable(in_channels, norm_layer)
            norm1 = AdaptiveNorm2dTrainable(out_channels, norm_layer)
        elif normalize:
            raise Exception('ResBlock: Incorrect `norm_layer` parameter')

        layers = []
        if normalize:
            layers.append(norm0)
        layers.append(activation(inplace=True))
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        layers.extend([
            #padding(1),
            #spectral_norm(
                shuffleConv(in_channels, out_channels, 3, 1, 0, bias=bias)#,
            #    eps=1e-4)
        ])
        if normalize:
            layers.append(norm1)
        layers.extend([
            activation(inplace=True),
            #padding(1),
            #spectral_norm(
                shuffleConv(out_channels, out_channels, 3, 1, 0, bias=bias)#,
            #    eps=1e-4)
        ])
        if downsample:
            layers.append(nn.AvgPool2d(2))
        self.block = nn.Sequential(*layers)

        self.skip = None
        if in_channels != out_channels or upsample or downsample:
            layers = []
            if upsample:
                layers.append(nn.Upsample(scale_factor=2))
            layers.append(
                #spectral_norm(
                shuffleConv(in_channels, out_channels, 1)#,
            #    eps=1e-4)
            )
            if downsample:
                layers.append(nn.AvgPool2d(2))
            self.skip = nn.Sequential(*layers)

    def forward(self, input):
        out = self.block(input)
        if self.skip is not None:
            output = out + self.skip(input)
        else:
            output = out + input
        return output
    
    

class ResBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups, 
                 resize_layer, norm_layer, activation):
        super(ResBlockV2, self).__init__()
        upsampling_layers = {
            'nearest': lambda: nn.Upsample(scale_factor=stride, mode='nearest')
        }
        downsampling_layers = {
            'avgpool': lambda: nn.AvgPool2d(stride)
        }
        norm_layers = {
            'bn': lambda num_features: SyncBatchNorm(num_features, momentum=1.0, eps=1e-4),
            'in': lambda num_features: nn.InstanceNorm2d(num_features, eps=1e-4, affine=True),
            'adabn': lambda num_features: AdaptiveNorm2d(num_features, 'bn'),
            'adain': lambda num_features: AdaptiveNorm2d(num_features, 'in')
        }
        normalize = norm_layer != 'none'
        bias = not normalize
        upsample = resize_layer in upsampling_layers
        downsample = resize_layer in downsampling_layers
        if normalize: 
            norm_layer = norm_layers[norm_layer]

        layers = []
        if normalize:
            layers.append(norm_layer(in_channels))
        layers.append(activation())
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        layers.extend([
            spectral_norm(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias),
                eps=1e-4)])
        if normalize:
            layers.append(norm_layer(out_channels))
        layers.extend([
            activation(),
            spectral_norm(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias),
                eps=1e-4)])
        if downsample:
            layers.append(nn.AvgPool2d(2))
        self.block = nn.Sequential(*layers)

        self.skip = None
        if in_channels != out_channels or upsample or downsample:
            layers = []
            if upsample:
                layers.append(nn.Upsample(scale_factor=2))
            layers.append(spectral_norm(
                nn.Conv2d(in_channels, out_channels, 1),
                eps=1e-4))
            if downsample:
                layers.append(nn.AvgPool2d(2))
            self.skip = nn.Sequential(*layers)

    def forward(self, input):
        out = self.block(input)
        if self.skip is not None:
            output = out + self.skip(input)
        else:
            output = out + input
        return output

class ResBlockV2Shuffle(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups, 
                 resize_layer, norm_layer, activation):
        super(ResBlockV2Shuffle, self).__init__()
        upsampling_layers = {
            'nearest': lambda: nn.Upsample(scale_factor=stride, mode='nearest')
        }
        downsampling_layers = {
            'avgpool': lambda: nn.AvgPool2d(stride)
        }
        norm_layers = {
            'bn': lambda num_features: SyncBatchNorm(num_features, momentum=1.0, eps=1e-4),
            'in': lambda num_features: nn.InstanceNorm2d(num_features, eps=1e-4, affine=True),
            'adabn': lambda num_features: AdaptiveNorm2d(num_features, 'bn'),
            'adain': lambda num_features: AdaptiveNorm2d(num_features, 'in')
        }
        normalize = norm_layer != 'none'
        bias = not normalize
        upsample = resize_layer in upsampling_layers
        downsample = resize_layer in downsampling_layers
        if normalize: 
            norm_layer = norm_layers[norm_layer]

        layers = []
        if normalize:
            layers.append(norm_layer(in_channels))
        layers.append(activation())
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        layers.extend([
            #spectral_norm(
                shuffleConv(in_channels, out_channels, 3, 1, 1, bias=bias)#,
            #    eps=1e-4)
        ])
        if normalize:
            layers.append(norm_layer(out_channels))
        layers.extend([
            activation(),
            #spectral_norm(
                shuffleConv(out_channels, out_channels, 3, 1, 1, bias=bias)#,
            #    eps=1e-4)
        ])
        if downsample:
            layers.append(nn.AvgPool2d(2))
        self.block = nn.Sequential(*layers)

        self.skip = None
        if in_channels != out_channels or upsample or downsample:
            layers = []
            if upsample:
                layers.append(nn.Upsample(scale_factor=2))
            layers.append(#spectral_norm(
                shuffleConv(in_channels, out_channels, 1)#,
            #    eps=1e-4)
            )
            if downsample:
                layers.append(nn.AvgPool2d(2))
            self.skip = nn.Sequential(*layers)

    def forward(self, input):
        out = self.block(input)
        if self.skip is not None:
            output = out + self.skip(input)
        else:
            output = out + input
        return output
    
    

class GatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_fun, kernel_size, stride=1, padding=0, bias=True):
        super(GatedBlock, self).__init__()
        self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                                  eps=1e-4)
        self.gate = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                                  eps=1e-4)
        self.act_fun = act_fun()
        self.gate_act_fun = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        out = self.act_fun(out)

        mask = self.gate(x)
        mask = self.gate_act_fun(mask)

        out_masked = out * mask
        return out_masked


class GatedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding, upsample, downsample,
                 norm_layer, activation=nn.ReLU):
        super(GatedResBlock, self).__init__()
        normalize = norm_layer != 'none'
        bias = not normalize

        if norm_layer == 'in':
            norm0 = nn.InstanceNorm2d(in_channels, eps=1e-4, affine=True)
            norm1 = nn.InstanceNorm2d(out_channels, eps=1e-4, affine=True)
        elif 'ada' in norm_layer:
            norm0 = AdaptiveNorm2d(in_channels, norm_layer)
            norm1 = AdaptiveNorm2d(out_channels, norm_layer)
        elif 'tra' in norm_layer:
            norm0 = AdaptiveNorm2dTrainable(in_channels, norm_layer)
            norm1 = AdaptiveNorm2dTrainable(out_channels, norm_layer)
        elif normalize:
            raise Exception('ResBlock: Incorrect `norm_layer` parameter')

        main_layers = []

        if normalize:
            main_layers.append(norm0)
        if upsample:
            main_layers.append(nn.Upsample(scale_factor=2))

        main_layers.extend([
            padding(1),
            GatedBlock(in_channels, out_channels, activation, 3, 1, 0, bias=bias)])

        if normalize:
            main_layers.append(norm1)
        main_layers.extend([
            padding(1),
            GatedBlock(out_channels, out_channels, activation, 3, 1, 0, bias=bias)])
        if downsample:
            main_layers.append(nn.AvgPool2d(2))

        self.main_pipe = nn.Sequential(*main_layers)

        self.skip_pipe = None
        if in_channels != out_channels or upsample or downsample:
            skip_layers = []
            
            if upsample:
                skip_layers.append(nn.Upsample(scale_factor=2))

            skip_layers.append(GatedBlock(in_channels, out_channels, activation, 1))

            if downsample:
                skip_layers.append(nn.AvgPool2d(2))
            self.skip_pipe = nn.Sequential(*skip_layers)

    def forward(self, input):
        mp_out = self.main_pipe(input)
        if self.skip_pipe is not None:
            output = mp_out + self.skip_pipe(input)
        else:
            output = mp_out + input
        return output


class ResBlockWithoutSpectralNorms(nn.Module):
    def __init__(self, in_channels, out_channels, padding, upsample, downsample,
                 norm_layer, activation=nn.ReLU):
        super(ResBlockWithoutSpectralNorms, self).__init__()
        normalize = norm_layer != 'none'
        bias = not normalize

        # if norm_layer == 'bn':
        #     # norm0 = SyncBatchNorm(in_channels, momentum=1.0, eps=1e-4)
        #     # norm1 = SyncBatchNorm(out_channels, momentum=1.0, eps=1e-4)
        #     pass
        if norm_layer == 'in':
            norm0 = nn.InstanceNorm2d(in_channels, eps=1e-4, affine=True)
            norm1 = nn.InstanceNorm2d(out_channels, eps=1e-4, affine=True)
        elif 'ada' in norm_layer:
            norm0 = AdaptiveNorm2d(in_channels, norm_layer)
            norm1 = AdaptiveNorm2d(out_channels, norm_layer)
        elif 'tra' in norm_layer:
            norm0 = AdaptiveNorm2dTrainable(in_channels, norm_layer)
            norm1 = AdaptiveNorm2dTrainable(out_channels, norm_layer)
        elif normalize:
            raise Exception('ResBlock: Incorrect `norm_layer` parameter')

        layers = []
        if normalize:
            layers.append(norm0)
        layers.append(activation(inplace=True))
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        layers.extend([
            padding(1),
            # spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, 1, 0, bias=bias)  # ,
            #    eps=1e-4)
        ])
        if normalize:
            layers.append(norm1)
        layers.extend([
            activation(inplace=True),
            padding(1),
            # spectral_norm(
            nn.Conv2d(out_channels, out_channels, 3, 1, 0, bias=bias)  # ,
            #    eps=1e-4)
        ])
        if downsample:
            layers.append(nn.AvgPool2d(2))
        self.block = nn.Sequential(*layers)

        self.skip = None
        if in_channels != out_channels or upsample or downsample:
            layers = []
            if upsample:
                layers.append(nn.Upsample(scale_factor=2))
            layers.append(  # spectral_norm(
                nn.Conv2d(in_channels, out_channels, 1)  # ,
                #    eps=1e-4)
            )
            if downsample:
                layers.append(nn.AvgPool2d(2))
            self.skip = nn.Sequential(*layers)

    def forward(self, input):
        out = self.block(input)
        if self.skip is not None:
            output = out + self.skip(input)
        else:
            output = out + input
        return output


class MobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding, upsample, downsample,
                 norm_layer, activation=nn.ReLU6, expansion_factor=6):
        super(MobileNetBlock, self).__init__()
        normalize = norm_layer != 'none'
        bias = not normalize

        conv0 = nn.Conv2d(in_channels, int(in_channels * expansion_factor), 1)
        dwise = nn.Conv2d(int(in_channels * expansion_factor), int(in_channels * expansion_factor), 3,
                          2 if downsample else 1, 1, groups=int(in_channels * expansion_factor))
        conv1 = nn.Conv2d(int(in_channels * expansion_factor), out_channels, 1)

        if norm_layer == 'bn':
            # norm0 = SyncBatchNorm(in_channels, momentum=1.0, eps=1e-4)
            # norm1 = SyncBatchNorm(out_channels, momentum=1.0, eps=1e-4)
            pass
        if 'in' in norm_layer:
            norm0 = nn.InstanceNorm2d(int(in_channels * expansion_factor), eps=1e-4, affine=True)
            norm1 = nn.InstanceNorm2d(int(in_channels * expansion_factor), eps=1e-4, affine=True)
            norm2 = nn.InstanceNorm2d(out_channels, eps=1e-4, affine=True)
        if 'ada' in norm_layer:
            norm2 = AdaptiveNorm2d(out_channels, norm_layer)
        elif 'tra' in norm_layer:
            norm2 = AdaptiveNorm2dTrainable(out_channels, norm_layer)

        # layers = [spectral_norm(conv0, eps=1e-4)]
        layers = [conv0]
        if normalize: layers.append(norm0)
        layers.append(activation(inplace=True))
        if upsample: layers.append(nn.Upsample(scale_factor=2))
        # layers.append(spectral_norm(dwise, eps=1e-4))
        layers.append(dwise)
        if normalize: layers.append(norm1)
        layers.extend([
            activation(inplace=True),
            # spectral_norm(
            conv1  # ,
            # eps=1e-4)
        ])
        if normalize: layers.append(norm2)
        self.block = nn.Sequential(*layers)

        self.skip = None
        if in_channels != out_channels or upsample or downsample:
            layers = []
            if upsample: layers.append(nn.Upsample(scale_factor=2))
            layers.append(
                # spectral_norm(
                nn.Conv2d(in_channels, out_channels, 1)  # ,
                #    eps=1e-4)
            )
            if downsample:
                layers.append(nn.AvgPool2d(2))
            self.skip = nn.Sequential(*layers)

    def forward(self, input):
        out = self.block(input)
        if self.skip is not None:
            output = out + self.skip(input)
        else:
            output = out + input
        return output


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(-1)

    def forward(self, input):
        b, c, h, w = input.shape
        query = self.query_conv(input).view(b, -1, h * w).permute(0, 2, 1)  # B x HW x C/8
        key = self.key_conv(input).view(b, -1, h * w)  # B x C/8 x HW
        energy = torch.bmm(query, key)  # B x HW x HW
        attention = self.softmax(energy)  # B x HW x HW
        value = self.value_conv(input).view(b, -1, h * w)  # B x C x HW

        out = torch.bmm(value, attention.permute(0, 2, 1)).view(b, c, h, w)
        output = self.gamma * out + input
        return output
