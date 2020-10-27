import torch
import torch.utils.data
import numpy as np
from .common import voxceleb, augmentation

from pathlib import Path

class Dataset:
    @staticmethod
    def get_args(parser):
        parser.add('--data_root', default='', type=Path)
        parser.add('--img_dir', default='Img', type=Path)
        parser.add('--kp_dir', default='landmarks', type=Path)
        parser.add('--draw_oval', default=True, action="store_bool")

        parser.add('--n_frames_for_encoder', default=8, type=int)

        parser = augmentation.get_args(parser)
        return parser

    @staticmethod
    def get_dataset(args, part):
        dirlist = voxceleb.get_part_data(args, part)

        loader = voxceleb.SampleLoader(
            args.data_root, img_dir=args.img_dir, kp_dir=args.kp_dir,
            draw_oval=args.draw_oval, deterministic=part != 'train')

        augmenter = augmentation.get_augmentation_seq(args)
        dataset = VoxCeleb2Dataset(
            dirlist, loader, args.inference, args.n_frames_for_encoder, args.image_size, augmenter)

        return dataset


class VoxCeleb2Dataset(voxceleb.VoxCeleb2Dataset):
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
            }
            dec_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in dec_ids]
            dec_dict = torch.utils.data.dataloader.default_collate(dec_dicts)
            del dec_dicts

            data_dict['target_rgbs'] = dec_dict['image']
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
            }
            dec_dicts = [self.loader.load_sample(path, i, self.imsize, **features_to_load) for i in dec_ids]
            dec_dict = torch.utils.data.dataloader.default_collate(dec_dicts)
            del dec_dicts

            data_dict['enc_stickmen'] = enc_dict['stickman']
            data_dict['enc_rgbs'] = enc_dict['image']
            data_dict['target_rgbs'] = dec_dict['image']
            data_dict['pose_input_rgbs'] = dec_dict['image']
            data_dict['dec_stickmen'] = dec_dict['stickman']
            data_dict['dec_keypoints'] = dec_dict['keypoints']

            target_dict['label'] = self.dirlist.index[index]

        if not self.augmenter.is_empty():
            raise NotImplementedError("Keypoints augmentation is NYI")
        data_dict['pose_input_rgbs'] = self.augmenter.augment_tensor(data_dict['pose_input_rgbs'])

        return data_dict, target_dict
