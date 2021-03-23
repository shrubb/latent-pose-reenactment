import numpy as np
import cv2
import torch
torch.set_grad_enabled(False)

from utils.crop_as_in_dataset import ImageWriter
from utils import utils

from pathlib import Path

from tqdm import tqdm

def string_to_valid_filename(x):
    return str(x).replace('/', '_')

if __name__ == '__main__':
    import logging, sys
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger('drive')

    import argparse
    arg_parser = argparse.ArgumentParser(
        description="Render 'puppeteering' videos, given a fine-tuned model and driving images.\n"
                    "Be careful: inputs have to be preprocessed by 'utils/preprocess_dataset.sh'.",
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('checkpoint_path', type=Path,
        help="Path to the *.pth checkpoint of a fine-tuned neural renderer model.")
    arg_parser.add_argument('data_root', type=Path,
        help="Driving images' source: \"root path\" that contains folders\n"
             "like 'images-cropped', 'segmentation-cropped-ffhq', or 'keypoints-cropped'.")
    arg_parser.add_argument('--images_paths', type=Path, nargs='+',
        help="Driving images' sources: paths to folders with driving images, relative to "
             "'`--data_root`/images-cropped' (note: here 'images-cropped' is the "
             "checkpoint's `args.img_dir`). Example: \"id01234/q2W3e4R5t6Y monalisa\".")
    arg_parser.add_argument('--destination', type=Path, required=True,
        help="Where to put the resulting videos: path to an existing folder.")
    args = arg_parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Will run on device '{device}'")

    # Initialize the model
    logger.info(f"Loading checkpoint from '{args.checkpoint_path}'")
    checkpoint_object = torch.load(args.checkpoint_path, map_location='cpu')

    import copy
    saved_args = copy.copy(checkpoint_object['args'])
    saved_args.finetune = True
    saved_args.inference = True
    saved_args.data_root = args.data_root
    saved_args.world_size = 1
    saved_args.num_workers = 1
    saved_args.batch_size = 1
    saved_args.device = device
    saved_args.bboxes_dir = Path("/non/existent/file")
    saved_args.prefetch_size = 4

    embedder, generator, _, running_averages, _, _, _ = \
        utils.load_model_from_checkpoint(checkpoint_object, saved_args)

    if 'embedder' in running_averages:
        embedder.load_state_dict(running_averages['embedder'])
    if 'generator' in running_averages:
        generator.load_state_dict(running_averages['generator'])

    embedder.train(not saved_args.set_eval_mode_in_test)
    generator.train(not saved_args.set_eval_mode_in_test)

    for driver_images_path in args.images_paths:
        # Initialize the data loader
        saved_args.val_split_path = driver_images_path
        from dataloaders.dataloader import Dataloader
        logger.info(f"Loading dataloader '{saved_args.dataloader}'")
        dataloader = Dataloader(saved_args.dataloader).get_dataloader(saved_args, part='val', phase='val')

        current_output_path = (args.destination / string_to_valid_filename(driver_images_path)).with_suffix('.MP4')
        current_output_path.parent.mkdir(parents=True, exist_ok=True)
        image_writer = ImageWriter.get_image_writer(current_output_path)

        for data_dict, _ in tqdm(dataloader):
            utils.dict_to_device(data_dict, device)

            embedder.get_pose_embedding(data_dict)
            generator(data_dict)

            def torch_to_opencv(image):
                image = image.permute(1,2,0).clamp_(0, 1).mul_(255).cpu().byte().numpy()
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR, dst=image)

            result = torch_to_opencv(data_dict['fake_rgbs'][0])
            pose_driver = torch_to_opencv(data_dict['pose_input_rgbs'][0, 0])

            frame_grid = np.concatenate((pose_driver, result), axis=1).astype(np.uint8)
            image_writer.add(frame_grid)
