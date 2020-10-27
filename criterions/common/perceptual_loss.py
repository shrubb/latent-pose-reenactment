import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(-1)


class PerceptualLoss(nn.Module):
    def __init__(self, weight, vgg_weights_dir, net='caffe', normalize_grad=False):
        super().__init__()
        self.weight = weight
        self.normalize_grad = normalize_grad

        if net == 'pytorch':
            model = torchvision.models.vgg19(pretrained=True).features

            mean = torch.tensor([0.485, 0.456, 0.406])
            std  = torch.tensor([0.229, 0.224, 0.225])

            num_layers = 30

        elif net == 'caffe':
            vgg_weights = torch.load(os.path.join(vgg_weights_dir, 'vgg19-d01eb7cb.pth'))

            map = {'classifier.6.weight': u'classifier.7.weight', 'classifier.6.bias': u'classifier.7.bias'}
            vgg_weights = OrderedDict([(map[k] if k in map else k, v) for k, v in vgg_weights.items()])

            model = torchvision.models.vgg19()
            model.classifier = nn.Sequential(Flatten(), *model.classifier._modules.values())

            model.load_state_dict(vgg_weights)

            model = model.features

            mean = torch.tensor([103.939, 116.779, 123.680]) / 255.
            std = torch.tensor([1., 1., 1.]) / 255.

            num_layers = 30

        elif net == 'face':
            # Load caffe weights for VGGFace, converted from
            # https://media.githubusercontent.com/media/yzhang559/vgg-face/master/VGG_FACE.caffemodel.pth
            # The base model is VGG16, not VGG19.
            model = torchvision.models.vgg16().features
            model.load_state_dict(torch.load(os.path.join(vgg_weights_dir, 'vgg_face_weights.pth')))

            mean = torch.tensor([103.939, 116.779, 123.680]) / 255.
            std = torch.tensor([1., 1., 1.]) / 255.

            num_layers = 30

        else:
            raise ValueError(f"Unknown type of PerceptualLoss: expected '{{pytorch,caffe,face}}', got '{net}'")

        self.register_buffer('mean', mean[None, :, None, None])
        self.register_buffer('std' ,  std[None, :, None, None])

        layers_avg_pooling = []

        for weights in model.parameters():
            weights.requires_grad = False

        for module in model.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                layers_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                layers_avg_pooling.append(module)

            if len(layers_avg_pooling) >= num_layers:
                break

        layers_avg_pooling = nn.Sequential(*layers_avg_pooling)

        self.model = layers_avg_pooling

    def normalize_inputs(self, x):
        return (x - self.mean) / self.std

    def forward(self, input, target):
        input = (input + 1) / 2
        target = (target.detach() + 1) / 2

        loss = 0

        features_input = self.normalize_inputs(input)
        features_target = self.normalize_inputs(target)

        for layer in self.model:
            features_input = layer(features_input)
            features_target = layer(features_target)

            if layer.__class__.__name__ == 'ReLU':
                if self.normalize_grad:
                    pass
                else:
                    loss = loss + F.l1_loss(features_input, features_target)

        return loss * self.weight
