import torch
from torch import nn

class Wrapper:
    @staticmethod
    def get_args(parser):
        pass

    @staticmethod
    def get_net(args):
        net = Embedder()
        return net.to(args.device)


class Embedder(nn.Module):
    def __init__(self):
        super().__init__()

    def enable_finetuning(self, _=None):
        pass

    def get_identity_embedding(self, _):
        pass

    def get_pose_embedding(self, _):
        pass

    def forward(self, data_dict):
        self.get_identity_embedding(data_dict)
        self.get_pose_embedding(data_dict)
