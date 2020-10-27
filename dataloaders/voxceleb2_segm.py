import torch
import torch.utils.data
import numpy as np
from .common import voxceleb, augmentation

from pathlib import Path
import cv2

import logging
logger = logging.getLogger('dataloader')

class Dataset:
    @staticmethod
    def get_args(parser):
        parser.add('--data_root', default='', type=Path)
        parser.add('--img_dir', default='Img', type=Path)
        parser.add('--kp_dir', default='landmarks', type=Path)
        parser.add('--segm_dir', default='segm', type=Path)
        parser.add('--draw_oval', default=True, action="store_bool")

        parser.add('--n_frames_for_encoder', default=8, type=int)

        parser = augmentation.get_args(parser)
        return parser

    @staticmethod
    def get_dataset(args, part):
        dirlist = voxceleb.get_part_data(args, part)

        loader = SampleLoader(
            args.data_root, img_dir=args.img_dir, kp_dir=args.kp_dir,
            draw_oval=args.draw_oval, segm_dir=args.segm_dir,
            deterministic=part != 'train')
        augmenter = augmentation.get_augmentation_seq(args)
        dataset = VoxCeleb2SegmDataset(
            dirlist, loader, args.inference, args.n_frames_for_encoder, args.image_size)

        return dataset


class SampleLoader(voxceleb.SampleLoader):
    """
        Extends `voxceleb.SampleLoader` with segmentation masks for each sample.
    """
    def __init__(
        self, data_root, img_dir=None, kp_dir=None,
        draw_oval=True, segm_dir=None, deterministic=False):

        super().__init__(
            data_root, img_dir, kp_dir,
            draw_oval=draw_oval, deterministic=deterministic)

        self.segm_dir = segm_dir

    def load_segm(self, path, i):
        segm_path = Path(self.data_root) / self.segm_dir / path / (i + '.png')
        segm_path_np = Path(self.data_root) / self.segm_dir / path / (i + '.png.npy')

        if segm_path.exists():
            segm = cv2.imread(str(segm_path))
            if segm is None:
                logger.critical(f"Couldn't load segmentation for {self.segm_dir}/{path}/{i}")
                segm = np.ones((1, 1, 3), dtype=np.float32)
            segm = segm[:, :, 1].astype(np.float32) / 255.
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
                    load_segmentation=False):
        retval = super().load_sample(
            path, i, imsize,
            load_image=load_image,
            load_stickman=load_stickman,
            load_keypoints=load_keypoints)

        if load_segmentation:
            segmentation = self.load_segm(path, i)
            segmentation = cv2.resize(segmentation, (imsize, imsize))
            segmentation = torch.tensor(segmentation)[None]
            segmentation = segmentation.expand((3,) + segmentation.shape[1:])
            retval['segmentation'] = segmentation

        return retval


class VoxCeleb2SegmDataset(voxceleb.VoxCeleb2Dataset):
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
                'load_stickman': True,
                'load_keypoints': True,
                'load_segmentation': not self.inference
            }
            dec_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in dec_ids]
            dec_dict = torch.utils.data.dataloader.default_collate(dec_dicts)
            del dec_dicts

            if not self.inference:
                data_dict['target_rgbs'] = dec_dict['image'] * dec_dict['segmentation']
                data_dict['real_segm'] = dec_dict['segmentation']

            data_dict['pose_input_rgbs'] = dec_dict['image']
            data_dict['dec_stickmen'] = dec_dict['stickman']
            data_dict['dec_keypoints'] = dec_dict['keypoints']
            # Also putting `enc_*` stuff for embedding pre-calculation before fune-tuning
            data_dict['enc_stickmen'] = dec_dict['stickman']
            data_dict['enc_rgbs'] = dec_dict['image']

            target_dict['label'] = 0
        else:
            # We are doing normal training, in which "one sample" is one video
            ids = self.loader.list_ids(path, self.n_frames_for_encoder+1)

            enc_ids = ids[:-1]
            dec_ids = ids[-1:]

            features_to_load = {
                'load_image': True,
                'load_stickman': True
            }
            enc_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in enc_ids]
            enc_dict = torch.utils.data.dataloader.default_collate(enc_dicts)
            del enc_dicts

            features_to_load = {
                'load_image': True,
                'load_stickman': True,
                'load_keypoints': True,
                'load_segmentation': not self.inference
            }
            dec_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in dec_ids]
            dec_dict = torch.utils.data.dataloader.default_collate(dec_dicts)
            del dec_dicts

            if not self.inference:
                data_dict['target_rgbs'] = dec_dict['image'] * dec_dict['segmentation']
                
                target_dict['real_segm'] = dec_dict['segmentation']
                
            data_dict['enc_stickmen'] = enc_dict['stickman']
            data_dict['enc_rgbs'] = enc_dict['image']
            data_dict['dec_keypoints'] = dec_dict['keypoints']
            data_dict['dec_stickmen'] = dec_dict['stickman']
            data_dict['pose_input_rgbs'] = dec_dict['image']

            target_dict['label'] = self.dirlist.index[index]

        if not self.augmenter.is_empty():
            raise NotImplementedError("Keypoints augmentation is NYI")
        data_dict['pose_input_rgbs'] = self.augmenter.augment_tensor(data_dict['pose_input_rgbs'])

        return data_dict, target_dict
