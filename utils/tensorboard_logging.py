import torch
import os.path
import datetime
import shutil
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import logging # standard python logging
logger = logging.getLogger('tensorboard_logging')

class MySummaryWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disk_space_available = True

    def check_if_disk_space_available(self):
        free_space_on_disk_MB = shutil.disk_usage(self.log_dir).free / 1024**2
        actually_disk_space_available = free_space_on_disk_MB > 1024
        if self.disk_space_available != actually_disk_space_available: # change of state, let's log it
            self.disk_space_available = actually_disk_space_available
            if actually_disk_space_available:
                logger.info("Disk space has freed up! Resuming tensorboard logging")
            else:
                logger.error("Stopping tensorboard logging: disk low on space")
        return actually_disk_space_available

    def add_scalar(self, *args):
        if self.check_if_disk_space_available():
            super().add_scalar(*args)

    def add_image(self, name, images_minibatch, captions, iteration):
        """
            images_minibatch: B x 3 x H x (k*W), float
            captions: 3 x h x (k*W), float
        """
        if self.check_if_disk_space_available():
            grid = make_grid(images_minibatch.detach().clamp(0, 1).data.cpu(), nrow=1)
            # Pad captions because `make_grid` also adds side padding
            captions = torch.nn.functional.pad(captions, () if len(images_minibatch) == 1 else (2,2))
            grid = torch.cat((captions, grid), dim=1) # Add a header with captions on top

            super().add_image(name, grid, iteration)


def get_postfix(args, default_args, args_to_ignore, delimiter='__'):
    s = []

    for arg in sorted(args.keys()):
        if not isinstance(arg, Path) and arg not in args_to_ignore and default_args[arg] != args[arg]:
            s += [f"{arg}^{args[arg]}"]

    return delimiter.join(s).replace('/', '+')  # .replace(';', '+')


def setup_logging(args, default_args, args_to_ignore, exp_name_use_date=True, tensorboard=True):
    if not args.experiment_name:
        args.experiment_name = get_postfix(vars(args), vars(default_args), args_to_ignore)
    
        if exp_name_use_date:
            time = datetime.datetime.now()
            args.experiment_name = time.strftime(f"%m-%d_%H-%M___{args.experiment_name}")

    save_dir = os.path.join(args.experiments_dir, args.experiment_name)
    os.makedirs(f'{save_dir}/checkpoints', exist_ok=True)

    writer = MySummaryWriter(save_dir, filename_suffix='_train') if tensorboard else None

    return save_dir, writer
