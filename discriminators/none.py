import torch
from torch import nn

class Wrapper:
    @staticmethod
    def get_args(parser):
        pass

    @staticmethod
    def get_net(args):
        return Discriminator().to(args.device)

    @staticmethod
    def get_optimizer(discriminator, args):
        return None

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.finetuning = False

    def enable_finetuning(self, _=None):
        self.finetuning = True

    def forward(self, _):
        pass
