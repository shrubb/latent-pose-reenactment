import torch
import torch.utils.data
import numpy as np
import cv2

from .common import voxceleb, augmentation

import copy
import math
from pathlib import Path

import logging
logger = logging.getLogger('dataloader')

class Dataset:
    @staticmethod
    def get_args(parser):
        parser.add('--data_root', default='', type=Path)
        parser.add('--img_dir', default='Img', type=Path)
        parser.add('--kp_dir', default='landmarks', type=Path)
        parser.add('--segm_dir', default='segm', type=Path)
        parser.add('--bboxes_dir', default="/Vol0/user/e.burkov/Shared/VoxCeleb2.1_bounding-boxes-SingleBBox.npy", type=Path)

        parser.add('--draw_oval', default=True, action="store_bool")

        parser.add('--n_frames_for_encoder', default=8, type=int)

        parser = augmentation.get_args(parser)
        return parser

    @staticmethod
    def get_dataset(args, part):
        dirlist = voxceleb.get_part_data(args, part)

        loader = SampleLoader(
            args.data_root, img_dir=args.img_dir, kp_dir=args.kp_dir,
            draw_oval=args.draw_oval, segm_dir=args.segm_dir, bboxes_dir=args.bboxes_dir,
            deterministic=part != 'train')

        augmenter = augmentation.get_augmentation_seq(args)
        dataset = VoxCeleb2SegmDataset(
            dirlist, loader, args.inference, args.n_frames_for_encoder, args.image_size, augmenter)

        return dataset


class SampleLoader(voxceleb.SampleLoader):
    """
        Extends `voxceleb.SampleLoader` with segmentation masks for each sample.
    """
    def __init__(
        self, data_root, img_dir=None, kp_dir=None,
        draw_oval=True, segm_dir=None, bboxes_dir=None,
        deterministic=False):

        super().__init__(
            data_root, img_dir, kp_dir,
            draw_oval=draw_oval, deterministic=deterministic)

        self.segm_dir = segm_dir
        
        try:
            self.bboxes = np.load(bboxes_dir, allow_pickle=True).item()
        except FileNotFoundError:
            self.bboxes = {}
            logger.warning(
                f"Could not find the '.npy' file with bboxes, will assume the "
                f"images are already cropped")

    def load_segm(self, path, i):
        segm_path = Path(self.data_root) / self.segm_dir / path / (i + '.png')
        segm_path_np = Path(self.data_root) / self.segm_dir / path / (i + '.png.npy')

        if segm_path.exists():
            # Pick the second channel:
            # with PGN, it denotes head+body; with Graphonomy, all 3 channels are identical.
            segm = cv2.imread(str(segm_path))[:, :, 1]
            if segm is None:
                logger.critical(f"Couldn't load segmentation for {self.segm_dir}/{path}/{i}")
                segm = np.ones((1, 1), dtype=np.uint8)
        elif segm_path_np.exists():
            segm = np.load(str(segm_path_np))
            segm = segm[:, :, 0]
        else:
            raise FileNotFoundError(f'Sample {segm_path} not found')

        return segm

    def load_sample(self, path, i, imsize,
                    load_image=False,
                    load_stickman=False,
                    load_keypoints=False,
                    load_bounding_box=False,
                    load_segmentation=False):
        retval = {}

        # Get bounding box
        try:
            identity, sequence = path.split('/')
            bbox = self.bboxes[identity][sequence][int(i)]
            l, t, r, b = (bbox / 256.0).tolist() # bbox coordinates in a space where image takes [0; 1] x [0; 1]

            # Make bbox square and scale it
            SCALE = 1.8

            center_x, center_y = (l + r) * 0.5, (t + b) * 0.5
            height, width = b - t, r - l
            new_box_size = max(height, width)
            l = center_x - new_box_size / 2 * SCALE
            r = center_x + new_box_size / 2 * SCALE
            t = center_y - new_box_size / 2 * SCALE
            b = center_y + new_box_size / 2 * SCALE
        except:
            # Could not find that bbox in `self.bboxes`, so assume this image is already cropped
            l, t, r, b = 0.0, 0.0, 1.0, 1.0

        # Load and crop image
        if load_image:
            image_original = self.load_rgb(path, i) # np.uint8, H x W x 3

            t_img, l_img, b_img, r_img = bbox_to_integer_coords(t, l, b, r, *image_original.shape[:2])
            # cv2.rectangle(image, (l_img, t_img), (r_img, b_img), (255,0,0), 2)

            # In VoxCeleb2.1, images have weird gray borders which are useless
            image = image_original[1:-1, 1:-1]
            t_img -= 1
            l_img -= 1
            r_img -= 1
            b_img -= 1

            # Crop
            image = crop_with_padding(image, t_img, l_img, b_img, r_img)

            # Resize to the target resolution
            image = cv2.resize(image, (imsize, imsize),
                interpolation=cv2.INTER_CUBIC if imsize > b_img - t_img else cv2.INTER_AREA)

            retval['image'] = torch.tensor(image.astype(np.float32) / 255.0).permute(2, 0, 1)

        if load_keypoints:
            assert load_image, \
                "Please also require the image if you need keypoints, because we need to know what are H and W"
            keypoints = self.load_keypoints(path, i)

            keypoints /= image_original.shape[1]
            keypoints -= [[l, t]]
            keypoints /= [[r-l, b-t]]

            retval['keypoints'] = torch.tensor(keypoints.astype(np.float32).flatten())

        if load_stickman:
            assert load_keypoints
            stickman = self.draw_stickman(image.shape[:2], keypoints * image.shape[0])
            retval['stickman'] = torch.tensor(stickman.astype(np.float32) / 255.0).permute(2, 0, 1)

        if load_segmentation:
            segmentation = self.load_segm(path, i)

            t_img, l_img, b_img, r_img = bbox_to_integer_coords(t, l, b, r, *segmentation.shape[:2])

            # In VoxCeleb2.1, images have weird gray borders which are useless
            segmentation = segmentation[1:-1, 1:-1]
            t_img -= 1
            l_img -= 1
            r_img -= 1
            b_img -= 1

            segmentation = crop_with_padding(segmentation, t_img, l_img, b_img, r_img, segmentation=True)

            segmentation = cv2.resize(segmentation, (imsize, imsize))
            segmentation = torch.tensor(segmentation.astype(np.float32) / 255.0)[None]
            segmentation = segmentation.expand((3,) + segmentation.shape[1:])
            retval['segmentation'] = segmentation

        return retval


class VoxCeleb2SegmDataset(voxceleb.VoxCeleb2Dataset):
    def __init__(self, dirlist, loader, inference, n_frames_for_encoder, imsize, augmenter):
        super().__init__(dirlist, loader, inference, n_frames_for_encoder, imsize, augmenter)

    def __getitem__(self, index):
        data_dict, target_dict = {}, {}
        row = self.dirlist.iloc[index]
        path = row['path']

        finetuning = 'file' in self.dirlist
        if finetuning:
            # We are doing fine-tuning (`self.dirlist` enumerates all images, not just identities)
            dec_ids = [row['file']]

            features_to_load = {
                'load_image': True,
                'load_segmentation': not self.inference
            }
            dec_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in dec_ids]
            dec_dict = torch.utils.data.dataloader.default_collate(dec_dicts)
            del dec_dicts

            data_dict['enc_rgbs'] = dec_dict['image']
            # data_dict['dec_keypoints'] = dec_dict['keypoints']
            data_dict['pose_input_rgbs'] = dec_dict['image']

            if not self.inference:
                data_dict['target_rgbs'] = dec_dict['image'] * dec_dict['segmentation']

                target_dict['real_segm'] = dec_dict['segmentation']

            target_dict['label'] = 0
        else:
            # We are doing normal training, in which "one sample" is one video
            ids = self.loader.list_ids(path, self.n_frames_for_encoder+1)

            enc_ids = ids[:-1]
            dec_ids = ids[-1:]

            features_to_load = {
                'load_image': True
            }
            enc_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in enc_ids]
            enc_dict = torch.utils.data.dataloader.default_collate(enc_dicts)
            del enc_dicts

            features_to_load = {
                'load_image': not self.inference,
                'load_segmentation': not self.inference
            }
            dec_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in dec_ids]
            dec_dict = torch.utils.data.dataloader.default_collate(dec_dicts)
            del dec_dicts

            data_dict['enc_rgbs'] = enc_dict['image']
            # data_dict['dec_keypoints'] = dec_dict['keypoints']
            data_dict['pose_input_rgbs'] = dec_dict['image']

            if not self.inference:
                data_dict['target_rgbs'] = dec_dict['image'] * dec_dict['segmentation']

                target_dict['real_segm'] = dec_dict['segmentation']

            target_dict['label'] = self.dirlist.index[index]

        if not self.inference:
            data_dict['pose_input_rgbs'], data_dict['target_rgbs'], target_dict['real_segm'] = \
                self.augmenter.augment_triplet(
                    data_dict['pose_input_rgbs'], data_dict['target_rgbs'], target_dict['real_segm'])

        return data_dict, target_dict

    def deterministic_(self, seed=0):
        return self.augmenter.deterministic_(seed)

def bbox_to_integer_coords(t, l, b, r, image_h, image_w):
    """
        t, l, b, r:
            float
            Bbox coordinates in a space where image takes [0; 1] x [0; 1].
        image_h, image_w:
            int

        return: t, l, b, r
            int
            Bbox coordinates in given image's pixel space.
            C-style indices (i.e. `b` and `r` are exclusive).
    """
    t *= image_h
    l *= image_h
    b *= image_h
    r *= image_h

    l, t = map(math.floor, (l, t))
    r, b = map(math.ceil, (r, b))

    # After rounding, make *exactly* square again
    b += (r - l) - (b - t)
    assert b - t == r - l

    # Make `r` and `b` C-style (=exclusive) indices
    r += 1
    b += 1
    return t, l, b, r

def crop_with_padding(image, t, l, b, r, segmentation=False):
    """
        image:
            numpy, H x W x 3
        t, l, b, r:
            int
        segmentation:
            bool
            Affects padding.

        return:
            numpy, (b-t) x (r-l) x 3
    """
    t_clamp, b_clamp = max(0, t), min(b, image.shape[0])
    l_clamp, r_clamp = max(0, l), min(r, image.shape[1])
    image = image[t_clamp:b_clamp, l_clamp:r_clamp]

    # If the bounding box went outside of the image, restore those areas by padding
    padding = [t_clamp - t, b - b_clamp, l_clamp - l, r - r_clamp]
    if sum(padding) == 0: # = if the bbox fully fit into image
        return image

    if segmentation:
        padding_top = [(x if i == 0 else 0) for i, x in enumerate(padding)]
        padding_others = [(x if i != 0 else 0) for i, x in enumerate(padding)]
        image = cv2.copyMakeBorder(image, *padding_others, cv2.BORDER_REPLICATE)
        image = cv2.copyMakeBorder(image, *padding_top, cv2.BORDER_CONSTANT)
    else:
        image = cv2.copyMakeBorder(image, *padding, cv2.BORDER_REFLECT101)
    assert image.shape[:2] == (b - t, r - l)

    # We will blur those padded areas
    h, w = image.shape[:2]
    y, x = map(np.float32, np.ogrid[:h, :w]) # meshgrids

    mask_l = np.full_like(x, np.inf) if padding[2] == 0 else (x / padding[2])
    mask_t = np.full_like(y, np.inf) if padding[0] == 0 else (y / padding[0])
    mask_r = np.full_like(x, np.inf) if padding[3] == 0 else ((w-1-x) / padding[3])
    mask_b = np.full_like(y, np.inf) if padding[1] == 0 else ((h-1-y) / padding[1])

    # The farther from the original image border, the more blur will be applied
    mask = np.maximum(
        1.0 - np.minimum(mask_l, mask_r),
        1.0 - np.minimum(mask_t, mask_b))

    # Do blur
    sigma = h * 0.016
    kernel_size = 0
    image_blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Now we'd like to do alpha blending math, so convert to float32
    def to_float32(x):
        x = x.astype(np.float32)
        x /= 255.0
        return x
    image = to_float32(image)
    image_blurred = to_float32(image_blurred)

    # Support 2-dimensional images (e.g. segmentation maps)
    if image.ndim < 3:
        image.shape += (1,)
        image_blurred.shape += (1,)
    mask.shape += (1,)

    # Replace padded areas with their blurred versions, and apply
    # some quickly fading blur to the inner part of the image
    image += (image_blurred - image) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)

    # Make blurred borders fade to edges
    if segmentation:
        fade_color = np.zeros_like(image)
        fade_color[:, :padding[2]] = 0.0
        fade_color[:, -padding[3]:] = 0.0
        mask = (1.0 - np.minimum(mask_l, mask_r))[:, :, None]
    else:
        fade_color = np.median(image, axis=(0,1))
    image += (fade_color - image) * np.clip(mask, 0.0, 1.0) 

    # Convert back to uint8 for interface consistency
    image *= 255.0
    image.round(out=image)
    image.clip(0, 255, out=image)
    image = image.astype(np.uint8)

    return image
