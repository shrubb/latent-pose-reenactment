import torch
from pathlib import Path
import shutil
import sys
import os
import cv2
import numpy as np


def make_visual(data, n_samples=2):
    output_images = []

    if 'enc_rgbs' in data:
        enc_rgb = data['enc_rgbs'][:n_samples, 0]
        output_images.append(("Identity src", enc_rgb))

    def add_one_driver(suffix, annotation):
        if 'dec_stickmen' + suffix in data:
            dec_stickmen = data['dec_stickmen' + suffix][:n_samples]
            if len(dec_stickmen.shape) > 4:
                dec_stickmen = dec_stickmen[:, 0]
            output_images.append((f"Pose src ({annotation})", dec_stickmen))
        elif 'pose_input_rgbs_cropped_voxceleb1' + suffix in data:
            real_rgb = data['pose_input_rgbs_cropped_voxceleb1' + suffix][:n_samples]
            if len(real_rgb.shape) > 4:
                real_rgb = real_rgb[:, 0]
            output_images.append((f"Pose src ({annotation})", real_rgb))
        elif 'target_rgbs' + suffix in data:
            real_rgb = data['target_rgbs' + suffix][:n_samples]
            if len(real_rgb.shape) > 4:
                real_rgb = real_rgb[:, 0]
            output_images.append((f"Pose target ({annotation})", real_rgb))
        if 'pose_input_rgbs' + suffix in data:
            real_rgb = data['pose_input_rgbs' + suffix][:n_samples]
            if len(real_rgb.shape) > 4:
                real_rgb = real_rgb[:, 0]
            output_images.append((f"Pose input ({annotation})", real_rgb))
        if 'fake_rgbs' + suffix in data:
            fake_rgb = data['fake_rgbs' + suffix][:n_samples]
            if len(fake_rgb.shape) > 4:
                fake_rgb = fake_rgb[:, 0]
            output_images.append(("Generator output", fake_rgb))

    add_one_driver('', 'same video')

    if 'real_segm' in data:
        real_segm = data['real_segm'][:n_samples]
        if len(real_segm.shape) > 4:
            real_segm = real_segm[:, 0]
        output_images.append(("True segmentation", real_segm))
    if 'fake_segm' in data:
        fake_segm = data['fake_segm'][:n_samples]
        if len(fake_segm.shape) > 4:
            fake_segm = fake_segm[:, 0]
        fake_segm = torch.cat([fake_segm]*3, dim=1)
        output_images.append(("Predicted segmentation", fake_segm))

    add_one_driver('_other_video', 'other video')
    add_one_driver('_other_person', 'other person')

    assert len(set(image.shape for _, image in output_images)) == 1 # all images are of same size
    with torch.no_grad():
        output_image_rows = torch.cat([image.cpu() for _, image in output_images], dim=3)

    captions_height = 38
    captions = [np.ones((captions_height, image.shape[3], 3), np.float32) for _, image in output_images]
    for caption_image, (text, _) in zip(captions, output_images):
        cv2.putText(caption_image, text, (1, captions_height-4), cv2.FONT_HERSHEY_PLAIN, 1.25, (0,0,0), 2)
    captions = np.concatenate(captions, axis=1)
    captions = torch.tensor(captions).permute(2,0,1).contiguous()

    return output_image_rows, captions


# TODO obsolete, remove
class Saver:
    def __init__(self, save_dir, save_fn='npz_per_batch', clean_dir=False):
        super(Saver, self).__init__()
        self.save_dir = Path(str(save_dir))
        self.need_save = True

        if clean_dir and os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

        os.makedirs(self.save_dir, exist_ok=True)

        self.save_fn = sys.modules[__name__].__dict__[save_fn]

    def save(self, epoch, **kwargs):
        self.save_fn(save_dir=self.save_dir, epoch=epoch, **kwargs)
