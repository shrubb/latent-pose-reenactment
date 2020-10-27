import torch
import torch.utils.data

import numpy as np
import cv2
import scipy

import random
from pathlib import Path

import logging # standard Python logging
logger = logging.getLogger('dataloader')

def get_part_data(args, part):
    """
    Load a list of VoxCeleb identities as a pandas dataframe (an identity is currently defined
    by a folder with images). Or, if `args.finetuning` is `True`, load a list of images for
    that identity.

    args:
        `argparse.Namespace` or any namespace
        Configuration arguments from 'train.py' launch.
    part:
        `str`
        'train' or 'val'.

    return:
    part_data:
        `pandas.DataFrame`, columns == ('path'[, 'file'])
    """
    logger = logging.getLogger(f"dataloaders.common.voxceleb.get_part_data ({part})")

    import pandas
    assert part in ('train', 'val'), "`.get_part_data()`'s `part` argument must be 'train' or 'val'"
    split_path = args.train_split_path if part == 'train' else args.val_split_path
    
    logger.info(f"Determining the '{part}' data source")

    def check_data_source_one_identity_path():
        logger.info(f"Checking if '{args.data_root / args.img_dir / split_path}' is a directory...")
        if (args.data_root / args.img_dir / split_path).is_dir():
            logger.info(f"Yes, it is; the only {part} identity will be '{split_path}'")
            return pandas.DataFrame({'path': [str(split_path)]})
        else:
            logger.info(f"No, it isn't")
            return None

    def check_data_source_identity_list_file():
        logger.info(f"Checking if '{split_path}' is a file...")
        if split_path.is_file():
            logger.info(f"Yes, it is; reading {part} identity list from it")
            return pandas.read_csv(split_path)
        else:
            logger.info(f"No, it isn't")
            return None

    def check_data_source_folder_with_identities():
        logger.info(f"Checking if '{args.data_root / args.img_dir}' is a directory...")
        if (args.data_root / args.img_dir).is_dir():
            identity_list = pandas.DataFrame({'path':
                sorted(str(x.relative_to(args.data_root)) for x in (args.data_root / args.img_dir).iterdir() if x.is_dir())
            })
            logger.info(f"Yes, it is; found {len(identity_list)} {part} identities in it")
            return identity_list
        else:
            logger.info(f"No, it isn't")
            return None

    check_data_source_functions = [
        check_data_source_one_identity_path,
        check_data_source_identity_list_file,
        check_data_source_folder_with_identities,
    ]
    for check_data_source in check_data_source_functions:
        identity_list = check_data_source()
        if identity_list is not None:
            break
    else:
        raise ValueError(
            f"Could not determine input data source, check `args.data_root`" \
            f", `args.img_dir` and `args.{part}_split_path")

    if args.finetune:
        if len(identity_list) > 1:
            raise NotImplementedError("Sorry, fine-tuning to multiple identities is not yet available")

        # In fine-tuning, let the dataframe hold paths to all images instead of identities
        from itertools import chain
        image_list = ((args.data_root / args.img_dir / x).iterdir() for x in identity_list['path'])
        image_list = sorted(chain(*image_list))
        logger.info(f"This dataset has {len(image_list)} images")

        args.num_labels = 1
        logger.info(f"Setting `args.num_labels` to 1 because we are fine-tuning or the model has been fine-tuned")

        retval = pandas.DataFrame({
            'path': [str(path.parent.relative_to(args.data_root / args.img_dir)) for path in image_list],
            'file': [path.stem                                                   for path in image_list]
        })
    else:
        # Make identity_list length exactly divisible by `world_size` so that epochs are synchronized among processes
        if args.checkpoint_path != "":
            logger.info(f"Truncating the identity list as in the checkpoint, to {args.num_labels} samples")
            identity_list = identity_list.iloc[:args.num_labels]
        else:
            if part == 'train':
                args.num_labels = len(identity_list)
                logger.info(f"Setting `args.num_labels` to {args.num_labels}")

        # Append some `identity_list`'s items to itself so that it's divisible by world size
        num_samples_to_add = (args.world_size - len(identity_list) % args.world_size) % args.world_size
        logger.info(
            f"Making dataset length divisible by world size: was {len(identity_list)}"
            f", became {len(identity_list) + num_samples_to_add}")
        retval = identity_list.append(identity_list.iloc[:num_samples_to_add])
    
    return retval

class SampleLoader:
    def __init__(
        self, data_root, img_dir=None, kp_dir=None,
        draw_oval=True, deterministic=False):

        self.data_root = data_root
        self.img_dir = img_dir
        self.kp_dir = kp_dir

        self.edges_parts, self.closed_parts, self.colors_parts = [], [], []

        # For drawing stickman    
        if draw_oval:
            self.edges_parts.append(list(range(0, 17)))
            self.closed_parts.append(False)
            self.colors_parts.append((255, 255, 255))

        self.edges_parts.extend([
            list(range(17, 22)),
            list(range(22, 27)),
            list(range(27, 31)),
            list(range(31, 36)),
            list(range(36, 42)),
            list(range(42, 48)),
            list(range(48, 60))])
        self.closed_parts.extend([False, False, False, False, True, True, True])
        self.colors_parts.extend([
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (0, 0, 255),
            (255, 0, 255),
            (0, 255, 255),
            (255, 255, 0)])

        self.deterministic = deterministic

    def list_ids(self, path, k):
        """
            path:
                str
                "{person_id}/{video_hash_string}/"
            k:
                int
                how many frames to sample from this video
        """
        full_path = self.data_root / self.img_dir / path
        id_list = list(full_path.iterdir())
        random_generator = random.Random(666) if self.deterministic else random

        while k > len(id_list):
            # just in case (unlikely) when we need to sample more frames than there are in this video
            id_list += list(full_path.iterdir())

        return [i.stem for i in random_generator.sample(id_list, k=k)]

    @staticmethod
    def calc_qsize(lm):
        lm_eye_left = lm[36: 42, :2]  # left-clockwise
        lm_eye_right = lm[42: 48, :2]  # left-clockwise
        lm_mouth_outer = lm[48: 60, :2]  # left-clockwise

        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        qsize = np.hypot(*x) * 2
        return qsize

    @staticmethod
    def pad_img(image, keypoints):
        if keypoints is None:
            return image, keypoints

        h, w, _ = image.shape
        qsize = SampleLoader.calc_qsize(keypoints)

        if w > h:
            pad_h = w - h
            pad_w = 0
        else:
            pad_h = 0
            pad_w = h - w

        image = np.pad(np.float32(image), ((pad_h, 0), (pad_w, 0), (0, 0)), 'reflect')
        keypoints[:, 1] += pad_h
        keypoints[:, 0] += pad_w

        h, w, _ = image.shape
        y, x, _ = np.ogrid[:h, :w, :1]

        pad = np.array([pad_w, pad_h, 0, 0]).astype(np.float32)
        pad[pad == 0] = 1e-10

        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        image += (scipy.ndimage.gaussian_filter(image, [blur, blur, 0]) - image) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        image += (np.median(image, axis=(0, 1)) - image) * np.clip(mask, 0.0, 1.0)

        return image, keypoints

    @staticmethod
    def make_image_square(image, keypoints):
        h, w, _ = image.shape

        if abs(h - w) > 1:
            image, keypoints = SampleLoader.pad_img(image, keypoints)

        if h - w == 1:
            image = image[:-1]
        elif h - w == -1:
            image = image[:, :-1]

        assert image.shape[0] == image.shape[1]
        return image, keypoints

    def load_rgb(self, path, i):
        img_path = self.data_root / self.img_dir / path / (i + '.jpg')
        image = cv2.imread(str(img_path))
        if image is None:
            logger.error(f"Couldn't load image {img_path}")
            image = np.zeros((1, 1), dtype=np.uint8)
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)

        return image

    def load_keypoints(self, path, i):
        keypoints_path = self.data_root / self.kp_dir / path / (i + '.npy')
        keypoints = np.load(keypoints_path)[:, :2]

        return keypoints

    def draw_stickman(self, image_shape, keypoints):
        stickman = np.zeros(image_shape + (3,), np.uint8)

        for edges, closed, color in zip(self.edges_parts, self.closed_parts, self.colors_parts):
            cv2.polylines(stickman, [keypoints.round()[edges].astype(np.int32)], closed, color, thickness=2)

        return stickman

    def load_sample(self, path, i, imsize,
                    load_image=False,
                    load_stickman=False,
                    load_keypoints=False):

        retval = {}

        if load_image:
            image = self.load_rgb(path, i)
            resize_ratio = imsize / image.shape[1]

        if load_stickman or load_keypoints:
            assert load_image, \
                "Please also require the image if you need keypoints, because we need to know what are H and W"
            keypoints = self.load_keypoints(path, i)

            keypoints *= resize_ratio

        if load_image:
            image = cv2.resize(image, (imsize, imsize),
                interpolation=cv2.INTER_CUBIC if resize_ratio > 1.0 else cv2.INTER_AREA)
            retval['image'] = torch.tensor(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        # image, keypoints = self.make_image_square(image, keypoints) # temporarily disabled

        if load_stickman:
            stickman = self.draw_stickman(image.shape[:2], keypoints)
            retval['stickman'] = torch.tensor(stickman.astype(np.float32) / 255.0).permute(2, 0, 1)

        if load_keypoints:
            retval['keypoints'] = torch.tensor(keypoints.astype(np.float32).flatten() / imsize)

        return retval


class VoxCeleb2Dataset(torch.utils.data.Dataset):
    def __init__(self, dirlist, loader, inference, n_frames_for_encoder, imsize, augmenter):
        self.loader = loader
        self.inference = inference
        self.dirlist = dirlist
        self.imsize = imsize
        self.n_frames_for_encoder = n_frames_for_encoder
        self.augmenter = augmenter

        # Temporary code for taking identity and pose from different people for visualization
        self.identity_to_labels = {}
        for record in self.dirlist.itertuples():
            identity = record.path[:7]
            if identity not in self.identity_to_labels:
                self.identity_to_labels[identity] = []
            self.identity_to_labels[identity].append(record.Index)

    # Temporary code for taking identity and pose from different people for visualization
    def get_other_sample_by_label(self, label, same_identity=False, deterministic=True):
        """
            label:
                `int`
                pandas index (="label") of a sample in dataset (e.g. the one
                from `data_dict['label']`).
            same_identity:
                `bool`
                See "return".
            deterministic:
                `bool`
                Return the index of the next sample, not a random one.

            return:
                `int`
                dataset (!) index of a random sample that has the SAME person but in a DIFFERENT
                video sequence. If `same_identity` is `False`, the person will also be DIFFERENT.
        """
        identity = self.dirlist.loc[label].path[:7]
        # All frames of the given person, including other videos
        labels_for_this_identity = self.identity_to_labels[identity]
        retval_index = 0
        if same_identity:
            while True:
                if not deterministic:
                    # Pick a random frame, but other than the given one
                    retval_label = random.choice(labels_for_this_identity)
                else:
                    # Pick next sutable frame, but other than the given one
                    retval_label = labels_for_this_identity[retval_index]
                    retval_index = retval_index + 1
                    
                if retval_label != label or len(labels_for_this_identity) == 1:
                    break

            return self.dirlist.index.get_loc(retval_label)
        else:
            retval_label = labels_for_this_identity[0]
            retval_index = self.dirlist.index.get_loc(retval_label)
            while True:
                if not deterministic:
                    # Pick a random frame, making sure there is other person in it
                    retval_index = random.randint(0, len(self) - 1)
                else:
                    # Pick next sutable frame, but other than the given one
                    if retval_index < self.dirlist.shape[0] - 1:
                        retval_index = retval_index + 1
                    else:
                        retval_index = 0
                        
                if self.dirlist.iloc[retval_index].path[:7] != identity or len(labels_for_this_identity) == len(self):
                    break
                
            return retval_index

    def __len__(self):
        return self.dirlist.shape[0]
