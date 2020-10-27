"""
Compute metrics (pose error and identity error) as defined in the "Neural Head Reenactment
with Latent Pose Descriptors" paper.

When running as a script, the reenactment results should be first obtained by
'batched_finetune.py' followed by 'batched_drive.py'.

Usage:
    First, change (or adapt your directories to) these paths below:
        `model=...`, `DATASET_ROOT`, `RESULTS_ROOT`, `DESCRIPTORS_GT_FILE`, `LANDMARKS_GT_FILE`
    Then:
    python3 compute_pose_identity_error.py <model-name>

Example usage:
    python3 compute_pose_identity_error.py MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_aug_noBottleneck_02492466

Args:
    <model-name>:
        A directory name where reenactment results are stored, as e.g.
        './puppeteering/VoxCeleb2_30Test/<model-name>/id03839_LhI_8AWX_Mg_identity/driving-results/*.mp4'.
"""
import face_alignment

from tqdm import tqdm

import numpy as np
import cv2

from pathlib import Path
import sys

############### ArcFace ###############

FACE_DESCRIPTOR_DIM = 512

arcface_model = None

def get_default_bbox(kind):
    """
    Get default rough estimate of face bounding box for `get_identity_descriptor()`.

    Args:
        kind:
            `str`
            Crop type that your model's pose encoder consumes.
            One of: 'ffhq', 'x2face', 'latentpose'.

    Returns:
        bbox:
            `tuple` of `int`, length == 4
            The bounding box in pixels. Defines how much pixels to clip from top, left,
            bottom and right in a 256 x 256 image.
    """
    if kind == 'ffhq':
        return (0, 30, 60, 30)
    elif kind == 'x2face':
        return (37, (37+45)//2, 45, (37+45)//2)
    elif kind == 'latentpose':
        return (42, (42+64)//2, 64, (42+64)//2)
    else:
        raise ValueError(f"Wrong crop type: {kind}")

def get_identity_descriptor(images, default_bbox):
    """
    Compute an identity vector (by the ArcFace face recognition system) for each image in `images`.

    Args:
        images:
            iterable of `numpy.ndarray`, dtype == uint8, shape == (256, 256, 3)
            Images to compute identity descriptors for.
        default_bbox:
            `tuple` of `int`, length == 4
            See `get_default_bbox()`.

    Returns:
        descriptors:
            `numpy.ndarray`, dtype == float32, shape == (`len(images)`, `FACE_DESCRIPTOR_DIM`)
        num_bad_images:
            int
            For how many images face detection failed.
    """
    global arcface_model

    # Load the model if it hasn't been loaded yet
    if arcface_model is None:
        from insightface import face_model

        arcface_model = face_model.FaceModel(
            image_size='112,112',
            model="/Vol0/user/e.burkov/Projects/insightface/models/model-r100-ii/model,0000",
            ga_model="",
            det=0,
            flip=1,
            threshold=1.24,
            gpu=0)

    num_bad_images = 0
    images_cropped = []

    for image in images:
        image_cropped = arcface_model.get_input(image)
        if image_cropped is None: # no faces found
            num_bad_images += 1
            t, l, b, r = default_bbox
            image_cropped = cv2.resize(image[t:256-b, l:256-r], (112, 112), interpolation=cv2.INTER_CUBIC)
            image_cropped = image_cropped.transpose(2, 0, 1)

        images_cropped.append(image_cropped)

    return arcface_model.get_feature(np.stack(images_cropped)), num_bad_images


############## Landmark detector ###############

MEAN_FACE = np.array([
    [74.0374984741211, 115.65937805175781],
    [74.81562805175781, 130.58021545410156],
    [77.2906265258789, 143.63853454589844],
    [80.5406265258789, 156.11041259765625],
    [85.6812515258789, 170.04791259765625],
    [93.36354064941406, 181.28541564941406],
    [101.20833587646484, 188.8718719482422],
    [110.51457977294922, 195.19479370117188],
    [126.53229522705078, 199.7687530517578],
    [142.9031219482422, 194.9875030517578],
    [154.76771545410156, 187.64999389648438],
    [163.98646545410156, 179.6666717529297],
    [172.2624969482422, 167.578125],
    [177.1437530517578, 152.93020629882812],
    [179.59478759765625, 139.87396240234375],
    [181.76145935058594, 125.9468765258789],
    [182.359375, 110.66458129882812],
    [84.17292022705078, 101.70625305175781],
    [89.2249984741211, 97.9437484741211],
    [96.4124984741211, 96.10104370117188],
    [103.30208587646484, 96.92916870117188],
    [109.55416870117188, 98.98958587646484],
    [135.68959045410156, 98.4749984741211],
    [142.27499389648438, 96.1500015258789],
    [149.71978759765625, 94.640625],
    [158.04896545410156, 95.68020629882812],
    [164.90728759765625, 99.32499694824219],
    [122.91041564941406, 114.76145935058594],
    [122.50416564941406, 125.12395477294922],
    [122.07604217529297, 134.3125],
    [122.16354370117188, 142.02915954589844],
    [115.19271087646484, 146.9250030517578],
    [118.640625, 148.04270935058594],
    [123.62187194824219, 149.28125],
    [128.79896545410156, 147.8489532470703],
    [132.8333282470703, 146.4479217529297],
    [94.09166717529297, 113.77291870117188],
    [98.35832977294922, 111.75],
    [104.53020477294922, 111.42916870117188],
    [110.55937194824219, 114.43645477294922],
    [105.203125, 116.39167022705078],
    [98.70207977294922, 116.40520477294922],
    [137.22084045410156, 113.53020477294922],
    [143.1770782470703, 110.64583587646484],
    [149.63645935058594, 110.56145477294922],
    [154.83749389648438, 112.0625],
    [149.82186889648438, 115.09479522705078],
    [142.86146545410156, 115.31041717529297],
    [107.09062194824219, 165.00416564941406],
    [112.30104064941406, 161.16354370117188],
    [119.99166870117188, 158.30313110351562],
    [124.18228912353516, 159.046875],
    [128.3802032470703, 158.02708435058594],
    [137.22084045410156, 160.6906280517578],
    [144.14688110351562, 164.3625030517578],
    [137.1770782470703, 170.67604064941406],
    [131.06353759765625, 174.26145935058594],
    [124.75104522705078, 175.1281280517578],
    [118.46145629882812, 174.7604217529297],
    [113.23645782470703, 171.27499389648438],
    [108.41666412353516, 164.7708282470703],
    [119.25729370117188, 163.55624389648438],
    [124.46979522705078, 163.3625030517578],
    [129.99583435058594, 163.53854370117188],
    [142.75416564941406, 164.22604370117188],
    [130.0520782470703, 167.13958740234375],
    [124.57083129882812, 167.7864532470703],
    [119.16666412353516, 167.3072967529297]], dtype=np.float32)

landmark_detector = None

def get_landmarks(image):
    """
    Compute 68 facial landmarks (2D ones!) by the Bulat et al. 2017 system from `image`.

    Args:
        image:
            `numpy.ndarray`, dtype == uint8, shape == (H, W, 3)

    Returns:
        landmarks:
            `numpy.ndarray`, dtype == float32, shape == (68, 2)
        success:
            bool
            False if no faces were detected.
    """
    global landmark_detector

    # Load the landmark detector if it hasn't been loaded yet
    if landmark_detector is None:
        landmark_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    landmarks = landmark_detector.get_landmarks(image)
    try:
        return landmarks[0], True
    except TypeError: # zero faces detected
        return MEAN_FACE, False

########################## List identities ###############################

# The ones used in the paper
IDENTITIES = [
    "id00061/cAT9aR8oFx0",
    "id00061/Df_m1slf_hY",
    "id00812/XoAi2n4S2wo",
    "id01106/B08yOvYMF7Y",
    "id01228/7qHTvs0VO68",
    "id01333/9kgJaduwKkY",
    "id01437/4lFDvxXzYWY",
    "id02057/s5VqJY7DDEE",
    "id02548/x2LUQEUXdz4",
    "id03127/uiRiyK8Qlic",
    "id03178/cCoNRuzAL-A",
    "id03178/fnARFfUwf2s",
    "id03524/GkvScYvOJ7o",
    "id03839/LhI_8AWX_Mg",
    "id03839/PUwanP-C5qg",
    "id03862/fsCqKQb9Rdg",
    "id04094/JUYMzfVp8zI",
    "id04950/PQEAck-3wcA",
    "id05459/3TI6dVmEwzw",
    "id05714/wFGNufaMbDY",
    "id06104/7UnGAS5-jpU",
    "id06811/KmvEwL3fP9Q",
    "id07312/h1dszoDi1E8",
    "id07663/54qlJ2HZ08s",
    "id07802/BfQUBDw7TiM",
    "id07868/JC0QT4oXh2Y",
    "id07961/464OHFffwjI",
    "id07961/hROZwL8pbGg",
    "id08149/vxBFGKGXSFA",
    "id08701/UeUyLqpLz70",
]

NUM_VIDEO_FRAMES = 32

########################## Define metrics ###############################

def identity_error(gt_descriptors, our_descriptors):
    assert gt_descriptors.shape == (len(IDENTITIES), FACE_DESCRIPTOR_DIM)
    assert our_descriptors.shape == (len(IDENTITIES), len(IDENTITIES), NUM_VIDEO_FRAMES, FACE_DESCRIPTOR_DIM)

    cosine_distances = (gt_descriptors[:, None, None] * our_descriptors).sum(-1).astype(np.float64)
    # Don't include self-driving
    for driver_idx in range(len(IDENTITIES)):
        cosine_distances[driver_idx][driver_idx] = 0

    return 1.0 - cosine_distances.sum() / (len(IDENTITIES) * (len(IDENTITIES) - 1) * NUM_VIDEO_FRAMES)

def pose_reconstruction_error(gt_landmarks, our_landmarks, apply_optimal_alignment=False):
    assert gt_landmarks.shape == (len(IDENTITIES), NUM_VIDEO_FRAMES, 68, 2)
    assert our_landmarks.shape == gt_landmarks.shape

    if apply_optimal_alignment:
        # The 3 variables here are scale, shift_x, shift_y.
        # Find a transform that optimizes || scale * our_landmarks + shift - gt_landmarks ||^2.
        alignments = np.empty(gt_landmarks.shape[:2] + (3,), dtype=np.float32)

        all_lhs = np.empty(gt_landmarks.shape + (3,), dtype=np.float64)
        all_lhs[:, :, :, :, 0] = our_landmarks
        all_lhs[:, :, :, 0, 1:] = [1, 0]
        all_lhs[:, :, :, 1, 1:] = [0, 1]
        all_lhs = all_lhs.reshape(len(IDENTITIES), NUM_VIDEO_FRAMES, -1, 3)

        all_rhs = gt_landmarks.astype(np.float64).reshape(len(IDENTITIES), NUM_VIDEO_FRAMES, -1)

        for i in range(len(IDENTITIES)):
            for f in range(NUM_VIDEO_FRAMES):
                alignments[i, f] = np.linalg.lstsq(all_lhs[i, f], all_rhs[i, f], rcond=None)[0]

        scale = alignments[:, :, 0, None, None] # `None` for proper broadcasting
        shift = alignments[:, :, None, 1:]
        our_landmarks = our_landmarks * scale + shift

    interocular = np.linalg.norm(gt_landmarks[:, :, 36] - gt_landmarks[:, :, 45], axis=-1).clip(min=1e-2)
    normalized_distances = np.linalg.norm(gt_landmarks - our_landmarks, axis=-1) / interocular[:, :, None]
    return normalized_distances.mean()


########################## The script ###############################

if __name__ == "__main__":

    # Where the "ground truth" driver/identity images are
    DATASET_ROOT = Path("/Vol0/user/e.burkov/Shared/VoxCeleb2_30TestIdentities")

    MODEL = sys.argv[1]
    # Where are cross-driving/self-driving outputs are, and where to cache the computed descriptors and landmarks
    RESULTS_ROOT = Path(f"puppeteering/VoxCeleb2_30Test/{MODEL}")
    assert RESULTS_ROOT.is_dir()

    ############  GT ArcFace  ##############

    if MODEL.startswith("Zakharov_0"):
        crop_type = 'ffhq'
    elif MODEL.startswith("X2Face_vanilla"):
        crop_type = 'x2face'
    else:
        crop_type = 'latentpose'
    print(f"Assuming the crop type is '{crop_type}'")
    DEFAULT_BBOX = get_default_bbox(crop_type)

    erase_background = not ('noSegm' in MODEL or MODEL.startswith("Zakharov_0") or MODEL.startswith("X2Face_vanilla"))

    if erase_background:
        DESCRIPTORS_GT_FILE = Path("puppeteering/VoxCeleb2_30Test/true_average_identity_descriptors_noBackground.npy")
    else:
        DESCRIPTORS_GT_FILE = Path("puppeteering/VoxCeleb2_30Test/true_average_identity_descriptors.npy")

    COMPUTE_GT_DESCRIPTORS = False # hardcode to `True` if you want to recompute
    try:
        gt_average_descriptors = np.load(DESCRIPTORS_GT_FILE)
        print(f"Loaded the cached target descriptors from {DESCRIPTORS_GT_FILE}")
    except FileNotFoundError:
        print(f"Could not load the target descriptors from {DESCRIPTORS_GT_FILE}")
        gt_average_descriptors = np.empty((len(IDENTITIES), FACE_DESCRIPTOR_DIM), dtype=np.float32)
        COMPUTE_GT_DESCRIPTORS = True

    if COMPUTE_GT_DESCRIPTORS:
        print(f"Recomputing target descriptors into {DESCRIPTORS_GT_FILE}")
        for identity, gt_average_descriptor in zip(tqdm(IDENTITIES), gt_average_descriptors):
            images_folder       = DATASET_ROOT /       'images-cropped' / identity / 'identity'
            segmentation_folder = DATASET_ROOT / 'segmentation-cropped' / identity / 'identity'
            identity_images = []
            for image_path in images_folder.iterdir():
                image = cv2.imread(str(image_path))
                if erase_background:
                    segmentation = cv2.imread(str(segmentation_folder / image_path.with_suffix('.png').name))
                    image = cv2.multiply(image, segmentation, dst=image, scale=1/255)

                identity_images.append(image)

            gt_descriptors, num_bad_images = get_identity_descriptor(identity_images, DEFAULT_BBOX)
            if num_bad_images > 0:
                print(f"===== WARNING: couldn't detect {num_bad_images} faces in {images_folder}")

            gt_average_descriptor[:] = gt_descriptors.mean(0)

        np.save(DESCRIPTORS_GT_FILE, gt_average_descriptors)

    ############# GT landmarks ##############

    def string_to_valid_filename(x):
        return x.replace('/', '_')

    COMPUTE_GT_LANDMARKS = False # hardcode to `True` if you want to recompute
    LANDMARKS_GT_FILE = Path("puppeteering/VoxCeleb2_30Test/target_landmarks.npy")
    try:
        gt_landmarks = np.load(LANDMARKS_GT_FILE)
        print(f"Loaded the cached target landmarks from {LANDMARKS_GT_FILE}")
    except FileNotFoundError:
        print(f"Couldn't load the cached target landmarks from {LANDMARKS_GT_FILE}")
        gt_landmarks = np.empty((len(IDENTITIES), NUM_VIDEO_FRAMES, 68, 2), dtype=np.float32)
        COMPUTE_GT_LANDMARKS = True

    if COMPUTE_GT_LANDMARKS:
        print(f"Recomputing target landmarks into {LANDMARKS_GT_FILE}")

        for identity_idx, identity in enumerate(IDENTITIES):
            images_folder = DATASET_ROOT / 'images-cropped' / identity / 'driver'
            for frame_idx, image_path in enumerate(sorted(images_folder.iterdir())):
                driver_image = cv2.imread(str(image_path))

                landmarks, success = get_landmarks(driver_image)
                if not success:
                    print(f"Failed to detect driver's landmarks in {image_path}")

                gt_landmarks[identity_idx, frame_idx] = landmarks

        np.save(LANDMARKS_GT_FILE, gt_landmarks)

    ############### cross-driving ArcFace and self-driving landmarks ###############

    our_landmarks = np.empty((len(IDENTITIES), NUM_VIDEO_FRAMES, 68, 2), dtype=np.float32)
    # identities x drivers x frames x 512
    our_descriptors = np.empty((len(IDENTITIES), len(IDENTITIES), NUM_VIDEO_FRAMES, FACE_DESCRIPTOR_DIM), dtype=np.float32)

    for identity_idx, identity in enumerate(IDENTITIES):
        IDENTITY_RESULTS_PATH = RESULTS_ROOT / (string_to_valid_filename(identity) + '_identity')

        (IDENTITY_RESULTS_PATH / "our_identity_descriptors").mkdir(parents=True, exist_ok=True)
        (IDENTITY_RESULTS_PATH / "our_landmarks").mkdir(parents=True, exist_ok=True)

        LANDMARKS_OUR_FILE = IDENTITY_RESULTS_PATH / "our_landmarks" / f"{string_to_valid_filename(identity)}.npy"
        COMPUTE_OUR_LANDMARKS = False # hardcode to `True` if you want to recompute
        try:
            our_landmarks[identity_idx] = np.load(LANDMARKS_OUR_FILE)
            print(f"Loaded the cached landmarks from {LANDMARKS_OUR_FILE}")
        except FileNotFoundError:
            print(f"Could not load our landmarks from {LANDMARKS_OUR_FILE}, recomputing")
            COMPUTE_OUR_LANDMARKS = True

        DESCRIPTORS_OUR_FILE = IDENTITY_RESULTS_PATH / "our_identity_descriptors" / f"{string_to_valid_filename(identity)}.npy"
        COMPUTE_OUR_DESCRIPTORS = False # hardcode to `True` if you want to recompute
        try:
            our_descriptors[identity_idx] = np.load(DESCRIPTORS_OUR_FILE)
            print(f"Loaded the cached face recognition descriptors from {DESCRIPTORS_OUR_FILE}")
        except FileNotFoundError:
            print(f"Could not load our descriptors from {DESCRIPTORS_OUR_FILE}, recomputing")
            COMPUTE_OUR_DESCRIPTORS = True

        if not COMPUTE_OUR_LANDMARKS and not COMPUTE_OUR_DESCRIPTORS:
            continue

        for driver_idx, driver in enumerate(tqdm(IDENTITIES)):
            video_path = IDENTITY_RESULTS_PATH / 'driving-results' / (string_to_valid_filename(driver) + '_driver.mp4')
            video_reader = cv2.VideoCapture(str(video_path))

            driver_images, reenacted_images = [], []

            for frame_idx in range(NUM_VIDEO_FRAMES):
                ok, image = video_reader.read()
                assert ok, video_path
                reenacted_images.append(image[:, 256:])

            if COMPUTE_OUR_DESCRIPTORS:
                identity_descriptors, num_bad_images = get_identity_descriptor(reenacted_images, DEFAULT_BBOX)
                if num_bad_images > 0:
                    print(f"===== WARNING: couldn't detect {num_bad_images} faces in {video_path}")

                our_descriptors[identity_idx, driver_idx] = identity_descriptors

            if COMPUTE_OUR_LANDMARKS and driver_idx == identity_idx:
                for frame_idx, reenacted_image in enumerate(reenacted_images):
                    landmarks, success = get_landmarks(reenacted_image)
                    if not success:
                        print(f"===== WARNING: failed to detect reenactment's landmarks in frame #{frame_idx} of {video_path}")

                    our_landmarks[identity_idx, frame_idx] = landmarks

        if COMPUTE_OUR_LANDMARKS:
            np.save(LANDMARKS_OUR_FILE, our_landmarks[identity_idx])
        if COMPUTE_OUR_DESCRIPTORS:
            np.save(DESCRIPTORS_OUR_FILE, our_descriptors[identity_idx])

    print(f"Identity error: {identity_error(gt_average_descriptors, our_descriptors)}")
    print(f"Pose reconstruction error: {pose_reconstruction_error(gt_landmarks, our_landmarks)}")
    print(f"Pose reconstruction error (with optimal alignment): {pose_reconstruction_error(gt_landmarks, our_landmarks, apply_optimal_alignment=True)}")
