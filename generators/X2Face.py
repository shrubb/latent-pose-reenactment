import torch
from torch import nn
from torch.nn.utils import spectral_norm
from generators.common import blocks

import logging # standard Python logging
logger = logging.getLogger('embedder')

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--X2Face_num_identity_images', type=int, default=3)

    @staticmethod
    def get_net(args):
        assert not args.weights_running_average, "Please set `weights_running_average: false` with X2Face"
        net = Generator(args.X2Face_num_identity_images)
        return net.to(args.device)

class Generator(nn.Module):
    def __init__(self, num_identity_images):
        super().__init__()

        self.identity_images = nn.Parameter(torch.empty(num_identity_images, 3, 256, 256))

        import sys
        X2FACE_ROOT_DIR = "embedders/X2Face"
        sys.path.append(f"{X2FACE_ROOT_DIR}/UnwrapMosaic/")
        try:
            from UnwrappedFace import UnwrappedFaceWeightedAverage
            state_dict = torch.load(
                f"{X2FACE_ROOT_DIR}/models/x2face_model_forpython3.pth", map_location='cpu')
        except (ImportError, FileNotFoundError):
            logger.critical(
                f"Please initialize submodules, then download 'x2face_model_forpython3.pth' from "
                f"http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/release_x2face_eccv_withpy3.zip"
                f" and put it into {X2FACE_ROOT_DIR}/models/")
            raise

        self.x2face_model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3, inner_nc=128)
        self.x2face_model.load_state_dict(state_dict['state_dict'])
        self.x2face_model.eval()

        # Forbid doing .train(), .eval() and .parameters()
        def train_noop(self, *args, **kwargs): pass
        def parameters_noop(self, *args, **kwargs): return []
        self.x2face_model.train = train_noop.__get__(self.x2face_model, nn.Module)
        self.x2face_model.parameters = parameters_noop.__get__(self.x2face_model, nn.Module)

        # Disable saving weights
        def state_dict_empty(self, *args, **kwargs): return {}
        self.x2face_model.state_dict = state_dict_empty.__get__(self.x2face_model, nn.Module)
        # Forbid loading weights after we have done that
        def _load_from_state_dict_noop(self, *args, **kwargs): pass
        for module in self.x2face_model.modules():
            module._load_from_state_dict = _load_from_state_dict_noop.__get__(module, nn.Module)

        self.finetuning = False

    @torch.no_grad()
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
        if data_dict is not None:
            self.identity_images = nn.Parameter(data_dict['enc_rgbs'][0]) # N x C x H x W

        self.finetuning = True

    @torch.no_grad()
    def forward(self, data_dict):
        batch_size = len(data_dict['pose_input_rgbs'])
        outputs = torch.empty_like(data_dict['pose_input_rgbs'][:, 0])

        for batch_idx in range(batch_size):
            # N x C x H x W
            identity_images = self.identity_images if self.finetuning else data_dict['enc_rgbs'][batch_idx]
            identity_images_list = []
            for identity_image in identity_images:
                identity_images_list.append(identity_image[None])
                
            # C x H x W
            pose_driver = data_dict['pose_input_rgbs'][batch_idx, 0]
            driver_images = pose_driver[None]

            result = self.x2face_model(driver_images, *identity_images_list)
            result = result.clamp(min=0, max=1)

            outputs[batch_idx].copy_(result[0])

        data_dict['fake_rgbs'] = outputs
        outputs.requires_grad_()