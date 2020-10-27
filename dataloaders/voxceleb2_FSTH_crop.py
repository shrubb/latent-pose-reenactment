import torch
import torch.utils.data
import numpy as np
import cv2

from .common import voxceleb, augmentation
from .voxceleb2 import VoxCeleb2Dataset, Dataset as ParentDataset

from pathlib import Path

class Dataset(ParentDataset):
    @staticmethod
    def get_dataset(args, part):
        dirlist = voxceleb.get_part_data(args, part)

        loader = SmallCropSampleLoader(
            args.data_root, img_dir=args.img_dir, kp_dir=args.kp_dir,
            draw_oval=args.draw_oval, deterministic=part != 'train')

        augmenter = augmentation.get_augmentation_seq(args)
        dataset = VoxCeleb2Dataset(
            dirlist, loader, args.inference, args.n_frames_for_encoder, args.image_size, augmenter)

        return dataset

class SmallCropSampleLoader(voxceleb.SampleLoader):
    def load_sample(self, path, i, imsize,
                    load_image=False,
                    load_stickman=False,
                    load_keypoints=False):

        retval = {}

        if load_image:
            image = self.load_rgb(path, i)
            original_image_size = image.shape

            cut_t, cut_b = 0.2, 1.0
            cut_l = (1.0 - (cut_b - cut_t)) / 2
            cut_r = 1.0 - cut_l

            cut_t = min(image.shape[0]-1, round(cut_t * image.shape[0]))
            cut_l = min(image.shape[1]-1, round(cut_l * image.shape[1]))
            cut_b = max(cut_t+1, round(cut_b * image.shape[0]))
            cut_r = max(cut_l+1, round(cut_r * image.shape[1]))

            image = image[cut_t:cut_b, cut_l:cut_r]

        if load_stickman or load_keypoints:
            assert load_image, \
                "Please also require the image if you need keypoints, because we need to know what are H and W"
            keypoints = self.load_keypoints(path, i)

            keypoints -= [[cut_l, cut_t]]
            keypoints *= [[imsize / (cut_r - cut_l), imsize / (cut_b - cut_t)]]

        if load_image:
            image = cv2.resize(image, (imsize, imsize),
                interpolation=cv2.INTER_CUBIC if imsize > image.shape[0] else cv2.INTER_AREA)
            retval['image'] = torch.tensor(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        # image, keypoints = self.make_image_square(image, keypoints) # temporarily disabled

        if load_stickman:
            stickman = self.draw_stickman(image.shape[:2], keypoints)
            retval['stickman'] = torch.tensor(stickman.astype(np.float32) / 255.0).permute(2, 0, 1)

        if load_keypoints:
            retval['keypoints'] = torch.tensor(keypoints.astype(np.float32).flatten() / imsize)

        return retval
