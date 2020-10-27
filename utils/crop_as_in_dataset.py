import torch
import torch.utils.data
import numpy as np
import cv2

import math
import os
from abc import ABC, abstractmethod

try:
    import face_alignment
    from face_alignment.detection.sfd import FaceDetector
except ImportError:
    raise ImportError(
        "Please install face alignment package from "
        "https://github.com/1adrianb/face-alignment")

def load_landmark_detector():
    return face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

def load_face_detector():
    return FaceDetector(device='cuda')

class FaceCropper(ABC):
    @abstractmethod
    def __init__(self, output_size=(256, 256)):
        """
            output_size
                tuple, (width x height)
        """
        pass

    @abstractmethod
    def crop_image(self, image, bbox=None, compute_landmarks=True):
        """
            image
                numpy.ndarray, np.uint8, H x W x 3, RGB
            bbox
                list, int, len == 4 or 5, LTRB
                If provided, don't run face detector.
            compute_landmarks:
                bool

            return:
                numpy.ndarray, np.uint8, `self.output_size` x 3 (e.g. 256 x 256 x 3), RGB
                    cropped image
                object
                    any crop auxiliary information (for now, facial landmarks)
        """
        pass

class FFHQFaceCropper(FaceCropper):
    """
        Yields "FFHQ-style" crops. Based on landmarks, which are detected
        using https://github.com/1adrianb/face-alignment.
    """
    def __init__(self, output_size=(256, 256)):
        self.landmark_detector = load_landmark_detector()
        self.output_size = output_size

    def crop_image(self, image, bbox=None, compute_landmarks=True):
        """
            image
                numpy.ndarray, np.uint8, H x W x 3, RGB
            bbox
                None
                No effect.
            compute_landmarks
                bool

            return:
                numpy.ndarray, np.uint8, `self.output_size` x 3 (e.g. 256 x 256 x 3), RGB
                    cropped image
                numpy.ndarray, np.float32, 68 x 3
                    cropped 3D landmarks
        """
        assert bbox is None, "NYI: custom bbox for FFHQFaceCropper"

        landmarks = self.landmark_detector.get_landmarks(image)
        try:
            landmarks = landmarks[0]
        except TypeError: # zero faces detected
            landmarks = np.random.rand(68, 3).astype(np.float32)

        image, landmarks = self.crop_from_landmarks(image, landmarks)
        
        h_resize_ratio = self.output_size[1] / image.shape[0]
        w_resize_ratio = self.output_size[0] / image.shape[1]
        landmarks[:, 0 ] *= h_resize_ratio
        landmarks[:, 1:] *= w_resize_ratio # scale Z too
        image = cv2.resize(image, self.output_size,
            interpolation=cv2.INTER_CUBIC if h_resize_ratio > 1.0 else cv2.INTER_AREA)

        return image, landmarks if compute_landmarks else None

    @staticmethod
    def crop_from_landmarks(image, landmarks, only_landmarks=False):
        """
            Crops an image as in VoxCeleb2 dataset, with blurred reflection
            padding, given pixel coordinates of 68 facial landmarks.

            image
                numpy.ndarray, np.uint8, H x W x 3
            landmarks
                numpy.ndarray, np.float32, 68 x {2|3}, float
            only_landmarks
                bool
                if True, image will not be cropped and returned

            return:
                numpy.ndarray, np.uint8, h x w x 3 (optional)
                    cropped image
                numpy.ndarray, np.float32, 68 x {2|3}
                    cropped landmarks
        """
        lm_chin          = landmarks[0  : 17, :2]  # left-right
        lm_eyebrow_left  = landmarks[17 : 22, :2]  # left-right
        lm_eyebrow_right = landmarks[22 : 27, :2]  # left-right
        lm_nose          = landmarks[27 : 31, :2]  # top-down
        lm_nostrils      = landmarks[31 : 36, :2]  # top-down
        lm_eye_left      = landmarks[36 : 42, :2]  # left-clockwise
        lm_eye_right     = landmarks[42 : 48, :2]  # left-clockwise
        lm_mouth_outer   = landmarks[48 : 60, :2]  # left-clockwise
        lm_mouth_inner   = landmarks[60 : 68, :2]  # left-clockwise

        lm_cropped = landmarks.copy()

        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x).item() * 2

        # Crop.
        border = max(round(qsize * 0.1), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (crop[0] - border, crop[1] - border, crop[2] + border, crop[3] + border)

        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - image.shape[1] + border, 0), max(pad[3] - image.shape[0] + border, 0))

        lm_cropped[:, 0] -= crop[0]
        lm_cropped[:, 1] -= crop[1]

        if not only_landmarks:
            def crop_from_bbox(img, bbox):
                """
                    bbox: tuple, (x1, y1, x2, y2)
                        x: horizontal, y: vertical, exclusive
                """
                x1, y1, x2, y2 = bbox
                if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
                    img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
                return img[y1:y2, x1:x2]

            def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
                img = cv2.copyMakeBorder(img,
                    -min(0, y1), max(y2 - img.shape[0], 0),
                    -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_REFLECT)
                y2 += -min(0, y1)
                y1 += -min(0, y1)
                x2 += -min(0, x1)
                x1 += -min(0, x1)
                return img, x1, x2, y1, y2

            image = crop_from_bbox(image, crop).astype(np.float32)

            h, w, _ = image.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            y, x = y.astype(np.float32), x.astype(np.float32)
            pad = np.array(pad, dtype=np.float32)
            pad[pad == 0] = 1e-10
            mask = np.maximum(1.0 - np.minimum(x / pad[0], (w-1-x) / pad[2]), 1.0 - np.minimum(y / pad[1], (h-1-y) / pad[3]))
            
            sigma = qsize * 0.02
            kernel_size = 0 #round(sigma * 4)
            image_blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma, borderType=cv2.BORDER_REFLECT)
            image += (image_blurred - image) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            image += (np.median(image, axis=(0,1)) - image) * np.clip(mask, 0.0, 1.0)
            image.round(out=image)
            image.clip(0, 255, out=image)
            image = image.astype(np.uint8)
            
        if only_landmarks:
            return lm_cropped
        else:
            return np.array(image), lm_cropped


class LatentPoseFaceCropper(FaceCropper):
    """
        Yields "latent pose style" crops. Based on face detections from the S^3FD detector.
    """
    def __init__(self, output_size=(256, 256)):
        """
            output_size
                tuple, (width x height)
        """
        self.face_detector = load_face_detector()
        self.landmark_detector = None # only loaded when `compute_landmarks=True` is requested
        self.output_size = output_size

    def crop_image(self, image, bbox=None, compute_landmarks=True):
        """
            image
                numpy.ndarray, np.uint8, H x W x 3, RGB
            bbox
                list, int, len == 4 or 5, LTRB
                If provided, don't run face detector.
            compute_landmarks
                bool

            return:
                numpy.ndarray, np.uint8, `self.output_size` x 3 (e.g. 256 x 256 x 3), RGB
                    cropped image
                numpy.ndarray, np.float32, 68 x 3
                    cropped 3D facial landmark coordinates (x, y, z)
        """
        if bbox is None:
            bboxes = self.detect_faces([image])[0]
            bbox = self.choose_one_detection(bboxes)[:4]

        if compute_landmarks:
            if self.landmark_detector is None:
                self.landmark_detector = load_landmark_detector()
            landmarks = self.landmark_detector.get_landmarks_from_image(image, [bbox])[0]

        # Make bbox square and scale it
        l, t, r, b = bbox
        SCALE = 1.8

        center_x, center_y = (l + r) * 0.5, (t + b) * 0.5
        height, width = b - t, r - l
        new_box_size = max(height, width)
        l = center_x - new_box_size / 2 * SCALE
        r = center_x + new_box_size / 2 * SCALE
        t = center_y - new_box_size / 2 * SCALE
        b = center_y + new_box_size / 2 * SCALE

        # Make floats integers
        l, t = map(math.floor, (l, t))
        r, b = map(math.ceil, (r, b))

        # After rounding, make *exactly* square again
        b += (r - l) - (b - t)
        assert b - t == r - l

        # Make `r` and `b` C-style (=exclusive) indices
        r += 1
        b += 1

        # Crop
        image_cropped = self.crop_with_padding(image, t, l, b, r)

        # "Crop" landmarks
        if compute_landmarks:
            landmarks[:, 0] -= l
            landmarks[:, 1] -= t

            h_resize_ratio = self.output_size[1] / image_cropped.shape[0]
            w_resize_ratio = self.output_size[0] / image_cropped.shape[1]
            landmarks[:, 0 ] *= h_resize_ratio
            landmarks[:, 1:] *= w_resize_ratio # scale Z too

        # Resize to the target resolution
        image_cropped = cv2.resize(image_cropped, self.output_size,
            interpolation=cv2.INTER_CUBIC if self.output_size[1] > bbox[3] - bbox[1] else cv2.INTER_AREA)

        return image_cropped, landmarks

    def detect_faces(self, images):
        """
            images
                list of numpy.ndarray, any dtype 0-255, H x W x 3, RGB
                OR
                torch.tensor, any dtype 0-255, B x 3 x H x W, RGB

                A batch of images.

            return:
                list of lists of lists of length 5
                Bounding boxes for each image in batch.
        """
        if type(images) is list:
            images = np.stack(images).transpose(0,3,1,2).astype(np.float32)
            images_torch = torch.tensor(images)
        else:
            assert torch.is_tensor(images)
            images_torch = images.to(torch.float32)

        return self.face_detector.detect_from_batch(images_torch.cuda())

    @staticmethod
    def choose_one_detection(frame_faces):
        """
            frame_faces
                list of lists of length 5
                several face detections from one image

            return:
                list of 5 floats
                one of the input detections: `(l, t, r, b, confidence)`
        """
        if len(frame_faces) == 0:
            retval = [0, 0, 200, 200, 0.0]
        elif len(frame_faces) == 1:
            retval = frame_faces[0]
        else:
            # sort by area, find the largest box
            largest_area, largest_idx = -1, -1
            for idx, face in enumerate(frame_faces):
                area = abs(face[2]-face[0]) * abs(face[1]-face[3])
                if area > largest_area:
                    largest_area = area
                    largest_idx = idx

            retval = frame_faces[largest_idx]

        return np.array(retval).tolist()

    @staticmethod
    def crop_with_padding(image, t, l, b, r, segmentation=False):
        """
            image:
                numpy, np.uint8, (H x W x 3) or (H x W)
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


VIDEO_EXTENSIONS = ('.avi', '.mpg', '.mov', '.mkv', '.mp4')
IMAGE_EXTENSIONS = ('.jpg', '.png')

class ImageReader(ABC):
    """
        An abstract iterator to read images from a folder, video, webcam, ...
    """
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __next__(self):
        """
            return:
                numpy, H x W x 3, uint8
        """
        pass

    def __getitem__(self, _):
        """
            A hack for using `ImageReader` in `torch.utils.data.DataLoader`.
        """
        return next(self)

    def __iter__(self):
        return self

    @staticmethod
    def get_image_reader(source):
        """
            source:
                `Path` or `str`
                See `crop_as_in_dataset.py`'s command line documentation for `SOURCE`.

            return:
                `ImageReader`
                A concrete `ImageReader` instance guessed from `source`.
        """
        source = str(source)

        if source.startswith('WEBCAM_'):
            return OpencvVideoCaptureReader(int(source[7:]))
        elif source[-4:].lower() in VIDEO_EXTENSIONS:
            return OpencvVideoCaptureReader(source)
        elif source[-4:].lower() in IMAGE_EXTENSIONS:
            return SingleImageReader(source)
        elif os.path.isdir(source):
            return FolderReader(source)
        else:
            raise ValueError(f"Invalid `source` argument: {source}")

class ImageWriter(ABC):
    """
        An abstract class to write images into (folder, video, screen, ...)
    """
    @abstractmethod
    def add(self, image, extra_data=None):
        """
            image:
                numpy.ndarray, H x W x 3, uint8
            extra_data:
                object
        """
        pass

    @staticmethod
    def get_image_writer(destination, fourcc=None, fps=None):
        """
            destination:
                `Path` or `str`
                See command line documentation for `DESTINATION`.
            fps:
                `float` (optional)

            return:
                `ImageWriter`
                An `ImageReader` instance guessed from `destination`.
        """
        destination = str(destination)

        if destination == 'SCREEN':
            return ScreenWriter()
        elif destination[-4:].lower() in IMAGE_EXTENSIONS:
            return SingleImageWriter(destination)
        elif destination[-4:].lower() in VIDEO_EXTENSIONS:
            return VideoWriter(destination, fourcc, fps)
        else:
            return FolderWriter(destination)

###### Image readers ######

class FolderReader(ImageReader):
    def __init__(self, path):
        self.path = str(path)
        self.files = sorted(os.listdir(self.path))
        self.index = 0

    def __len__(self):
        return len(self.files)

    def __next__(self):
        if self.index == len(self.files):
            raise StopIteration

        image = cv2.imread(os.path.join(self.path, self.files[self.index]))

        self.index += 1
        return image

class OpencvVideoCaptureReader(ImageReader):
    def __init__(self, source):
        self.video_capture = cv2.VideoCapture(str(source))
        assert self.video_capture.isOpened()

    def __len__(self):
        retval = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return None if retval < 0 else retval

    def __next__(self):
        success, image = self.video_capture.read()
        if not success:
            raise StopIteration
        else:
            return image

class SingleImageReader(ImageReader):
    def __init__(self, path):
        self.image = cv2.imread(str(path))

    def __len__(self):
        return 1

    def __next__(self):
        if self.image is None:
            raise StopIteration

        try:
            return self.image
        finally:
            self.image = None

###### Image writers ######

class FolderWriter(ImageWriter):
    def __init__(self, path):
        self.path = str(path)
        if os.path.exists(path):
            num_files = len(os.listdir(path))
            print(f"WARNING: {path} already exists, contains {num_files} files")

        os.makedirs(path, exist_ok=True)

        self.index = 0

    def add(self, image, extra_data=None):
        cv2.imwrite(os.path.join(self.path, '%05d.jpg' % self.index), image)

        if extra_data is not None:
            np.save(os.path.join(self.path, '%05d.npy' % self.index), extra_data)

        self.index += 1

class VideoWriter(ImageWriter):
    def __init__(self, path, fourcc=None, fps=None):
        """
            path: str
                Where to save the video. Extension matters (see below).
            fourcc: str, length 4 (optional)
                Codec to use.
                - 'MJPG' with ".avi" extension is very safe platform agnostic, works everywhere.
                - 'avc1' or 'mp4v' with ".mp4" extension, on the other hand,
                  is good for sending to Telegram. Be careful: this most likely won't work
                  with pip's `opencv-python`, but you can use ffmpeg to convert to manually,
                  e.g. `ffmpeg -i input.avi -vcodec libx264 -f mp4 output.mp4`.
            fps: float (optional)
                Output video framerate. Default: 25.
        """
        self.path = str(path)

        if fourcc is None:
            default_codecs = {
                '.avi': 'MJPG',
                '.mp4': 'avc1',
            }
            fourcc = default_codecs.get(self.path[-4:], 'XVID')
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)

        self.fps = 25.0 if fps is None else fps
        self.video_writer = None

    def add(self, image, extra_data=None):
        if self.video_writer is None:
            self.video_writer = cv2.VideoWriter(
                self.path, self.fourcc, self.fps, image.shape[1::-1])
            assert self.video_writer.isOpened(), "Couldn't initialize video writer"

        self.video_writer.write(image)

class SingleImageWriter(ImageWriter):
    def __init__(self, path):
        self.path = str(path)

    def add(self, image, extra_data=None):
        cv2.imwrite(self.path, image)

        if extra_data:
            np.save(os.path.splitext(self.path)[0] + '.npy', extra_data)

class ScreenWriter(ImageWriter):
    def add(self, image):
        cv2.imshow('Cropped image', image)
        cv2.waitKey(1)


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="Crop images as in training dataset",
        formatter_class=argparse.RawTextHelpFormatter)
    
    arg_parser.add_argument('source', metavar='SOURCE', type=str,
        help="Where to take images from: can be\n"
             "- path to a folder with images,\n"
             "- path to a single image,\n"
             "- path to a video file,\n"
             "- 'WEBCAM_`N`', N=0,1,2... .")
    arg_parser.add_argument('destination', metavar='DESTINATION', type=str,
        help="Where to put cropped images: can be\n"
             "- path to a non-existent folder (will be created and filled with images),\n"
             "- path to a maybe existing image (guessed by extension),\n"
             "- path to a maybe existing video file (guessed by extension),\n"
             "- 'SCREEN'.")
    arg_parser.add_argument('--crop-style', type=str, default='latentpose', choices=['ffhq', 'latentpose'],
        help="Which crop style to use.")
    arg_parser.add_argument('--image-size', type=int, default=256,
        help="Size of square output images.")
    arg_parser.add_argument('--save-extra-data', action='store_true',
        help="If set, will save '.npy' files with keypoints alongside.")
    args = arg_parser.parse_args()

    # Initialize image reader
    image_reader = ImageReader.get_image_reader(args.source)
    image_loader = torch.utils.data.DataLoader(image_reader,
        num_workers=1 if isinstance(image_reader, FolderReader) else 0)

    # Initialize image writer
    fps = None
    if isinstance(image_reader, OpencvVideoCaptureReader):
        fps = image_reader.video_capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = None
    image_writer = ImageWriter.get_image_writer(args.destination, fps=fps)

    # Initialize cropper
    ChosenFaceCropper = {
        'ffhq':       FFHQFaceCropper,
        'latentpose': LatentPoseFaceCropper,
    }[args.crop_style]
    cropper = ChosenFaceCropper((args.image_size, args.image_size))

    # Main loop
    from tqdm import tqdm
    for input_image in tqdm(image_loader):
        input_image = input_image[0].numpy()

        if max(input_image.shape) > 1152:
            resize_ratio = 1152 / max(input_image.shape)
            input_image = cv2.resize(input_image, dsize=None, fx=resize_ratio, fy=resize_ratio)

        image_cropped, extra_data = cropper.crop_image(input_image)
        if not args.save_extra_data:
            extra_data = None
        image_writer.add(image_cropped, extra_data)
