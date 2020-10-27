import torch
from torch import nn
from torch.nn.utils import spectral_norm
from generators.common import blocks


class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--average_function', type=str, default='sum', help='sum|max')

    @staticmethod
    def get_net(args):
        net = Embedder(
            args.embed_channels, args.pose_embedding_size, args.average_function)
        return net.to(args.device)


class Embedder(nn.Module):
    def __init__(self, identity_embedding_size, pose_embedding_size, average_function):
        super().__init__()

        self.identity_embedding_size = identity_embedding_size
        self.pose_embedding_size = pose_embedding_size

        import torchvision
        self.identity_encoder = torchvision.models.resnext50_32x4d(num_classes=identity_embedding_size)
        self.pose_encoder = torchvision.models.mobilenet_v2(num_classes=pose_embedding_size)

        self.average_function = average_function

        self.finetuning = False

    def enable_finetuning(self, data_dict=None):
        self.finetuning = True

    def get_identity_embedding(self, data_dict):
        inputs = data_dict['enc_rgbs']

        batch_size, num_faces, c, h, w = inputs.shape

        inputs = inputs.view(-1, c, h, w)
        identity_embeddings = self.identity_encoder(inputs).view(batch_size, num_faces, -1)
        assert identity_embeddings.shape[2] == self.identity_embedding_size

        if self.average_function == 'sum':
            identity_embeddings_aggregated = identity_embeddings.mean(1)
        elif self.average_function == 'max':
            identity_embeddings_aggregated = identity_embeddings.max(1)[0]
        else:
            raise ValueError("Incorrect `average_function` argument, expected `sum` or `max`")

        data_dict['embeds'] = identity_embeddings_aggregated
        data_dict['embeds_elemwise'] = identity_embeddings

    def get_pose_embedding(self, data_dict):
        x = data_dict['pose_input_rgbs'][:, 0]
        data_dict['pose_embedding'] = self.pose_encoder(x)

    def forward(self, data_dict):
        if not self.finetuning:
            self.get_identity_embedding(data_dict)
        self.get_pose_embedding(data_dict)
