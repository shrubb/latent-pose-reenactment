import torch
from criterions.common.perceptual_loss import PerceptualLoss

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--idt_embed_weight', type=float, default=2e-3)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.idt_embed_weight, args.vgg_weights_dir)
        return criterion.to(args.device)

class Criterion(torch.nn.Module):
    def __init__(self, idt_embed_weight, vgg_weights_dir):
        super().__init__()
        self.idt_embed_crit = PerceptualLoss(idt_embed_weight, vgg_weights_dir, net='face').eval()

    def forward(self, data_dict):
        fake_rgb = data_dict['fake_rgbs']
        real_rgb = data_dict['target_rgbs']

        if len(fake_rgb.shape) > 4:
            fake_rgb = fake_rgb[:, 0]

        if len(real_rgb.shape) > 4:
            real_rgb = real_rgb[:, 0]

        if 'dec_keypoints' in data_dict:
            keypoints = data_dict['dec_keypoints']

            bboxes_estimate = compute_bboxes_from_keypoints(keypoints)

            # convert bboxes from [0; 1] to pixel coordinates
            h, w = real_rgb.shape[2:]
            bboxes_estimate[:, 0:2] *= h
            bboxes_estimate[:, 2:4] *= w
        else:
            crop_factor = 1 / 1.8
            h, w = real_rgb.shape[2:]

            t = h * (1 - crop_factor) / 2
            l = w * (1 - crop_factor) / 2
            b = h - t
            r = w - l
            
            bboxes_estimate = torch.empty((1, 4), dtype=torch.float32, device=real_rgb.device)
            bboxes_estimate[0].copy_(torch.tensor([t, b, l, r]))
            bboxes_estimate = bboxes_estimate.expand(len(real_rgb), 4)

        fake_rgb_cropped = crop_and_resize(fake_rgb, bboxes_estimate)
        real_rgb_cropped = crop_and_resize(real_rgb, bboxes_estimate)

        loss_G_dict = {}
        loss_G_dict['VGGFace'] = self.idt_embed_crit(fake_rgb_cropped, real_rgb_cropped)
        return loss_G_dict

def crop_and_resize(images, bboxes, target_size=None):
    """
    images: B x C x H x W
    bboxes: B x 4; [t, b, l, r], in pixel coordinates
    target_size (optional): tuple (h, w)

    return value: B x C x h x w

    Crop i-th image using i-th bounding box, then resize all crops to the
    desired shape (default is the original images' size, H x W).
    """
    t, b, l, r = bboxes.t().float()
    batch_size, num_channels, h, w = images.shape

    affine_matrix = torch.zeros(batch_size, 2, 3, dtype=torch.float32, device=images.device)
    affine_matrix[:, 0, 0] = (r-l) / w
    affine_matrix[:, 1, 1] = (b-t) / h
    affine_matrix[:, 0, 2] = (l+r) / w - 1
    affine_matrix[:, 1, 2] = (t+b) / h - 1

    output_shape = (batch_size, num_channels) + (target_size or (h, w))
    try:
        grid = torch.affine_grid_generator(affine_matrix, output_shape, False)
    except TypeError: # PyTorch < 1.4.0
        grid = torch.affine_grid_generator(affine_matrix, output_shape)
    return torch.nn.functional.grid_sample(images, grid, 'bilinear', 'reflection')

def compute_bboxes_from_keypoints(keypoints):
    """
    keypoints: B x 68*2

    return value: B x 4 (t, b, l, r)

    Compute a very rough bounding box approximate from 68 keypoints.
    """
    x, y = keypoints.float().view(-1, 68, 2).transpose(0, 2)

    face_height = y[8] - y[27]
    b = y[8] + face_height * 0.2
    t = y[27] - face_height * 0.47

    midpoint_x = (x.min() + x.max()) / 2
    half_height = (b - t) * 0.5
    l = midpoint_x - half_height
    r = midpoint_x + half_height

    return torch.stack([t, b, l, r], dim=1)
