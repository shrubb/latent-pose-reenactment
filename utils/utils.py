import importlib
import logging
import os
import random
import time
from argparse import Namespace
from collections import defaultdict
from typing import List

import cv2
import numpy as np
import torch
import yamlenv


def setup(args):
    logger = logging.getLogger('utils.setup')

    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    os.environ['OMP_NUM_THREADS'] = '1'

    if args.random_seed is None:
        args.random_seed = int(time.time() * 2)

    logger.info(f"Random Seed: {args.random_seed}")
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if args.device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.random_seed)


def dict_to_device(d, device):
    for key in d:
        if torch.is_tensor(d[key]):
            d[key] = d[key].to(device)


def get_args_and_modules(parser, use_checkpoint_args=True, custom_args={}):
    """
        Load modules ("embedder", "generator", "discriminator", "runner", "dataloader", criteria)
        and initialize program arguments based on various sources of those arguments. Optionally,
        also load a checkpoint file.

        The preference (i.e. the inverse resolution order) is:
            1. command line
            2. `custom_args`
            3. .yaml file
            4. `args` from a checkpoint file
            5. argparse defaults

        parser:
            `argparse.ArgumentParser`
        use_checkpoint_args:
            `bool`
            If `True`: check if `--checkpoint_path` is defined either
            in .yaml file or on the command line; if yes, use
            `saved_args` from that checkpoint.
        custom_args:
            `dict`
            Defines any custom default values for `parser`'s parameters.

        return:
            args:
                namespace
            <deprecated>:
                namespace
            m:
                `dict`
                A mapping from module names (e.g. 'generator') to actual loaded modules
            checkpoint_object:
                `dict` or `None`
                If `use_checkpoint_args` was `True`, the return value of
                `torch.load(args.checkpoint_path, map_location='cpu')`.
                Otherwise, `None`.
    """
    logger = logging.getLogger('utils.get_args_and_modules')

    # Parse arguments supplied on the command line
    # (and in `custom_args`) to obtain "--config_name" value
    parser.set_defaults(**custom_args)
    args, _ = parser.parse_known_args()

    # Read the .yaml config
    try:
        if args.config_name == '':
            logger.warning(f"Not using any .yaml config file")
            config_args = {}
        else:
            config_args = load_config_file(args.config_name)
    except FileNotFoundError:
        logger.warning(f"Could not load config {args.config_name}")
        config_args = {}

    # Let the parser update `args` with values from there.
    # We do this to obtain the "--checkpoint_path" value
    parser.set_defaults(**config_args)
    parser.set_defaults(**custom_args)
    args, _ = parser.parse_known_args()

    # If "--checkpoint_path" is defined, load the checkpoint to merge its args
    if use_checkpoint_args:
        if args.checkpoint_path:
            logger.info(f"Loading checkpoint file {args.checkpoint_path}")
            checkpoint_object = torch.load(args.checkpoint_path, map_location='cpu')
            checkpoint_args = vars(checkpoint_object['args'])
        else:
            logger.info(f"`args.checkpoint_path` isn't defined, so not using args from a checkpoint")
            checkpoint_object, checkpoint_args = None, {}
    else:
        checkpoint_object, checkpoint_args = None, {}

    # Go through the config resolution order (see docstring), but just to determine
    # module names, i.e. the following config arguments:
    # "embedder", "generator", "discriminator", "runner", "dataloader", "criterions"
    parser.set_defaults(**checkpoint_args)
    parser.set_defaults(**config_args)
    parser.set_defaults(**custom_args)
    args, _ = parser.parse_known_args()

    # Now when the module names are known, load them and update the argument parser accordingly,
    # since those modules may define their own variables with their own argument parsers:
    m = {}

    m['generator'] = load_module('generators', args.generator).Wrapper
    m['generator'].get_args(parser)

    m['embedder'] = load_module('embedders', args.embedder).Wrapper
    m['embedder'].get_args(parser)

    m['runner'] = load_module('runners', args.runner)
    m['runner'].get_args(parser)

    m['discriminator'] = load_module('discriminators', args.discriminator).Wrapper
    m['discriminator'].get_args(parser)

    m['criterion_list'] = load_wrappers_for_module_list(args.criterions, 'criterions')
    for crit in m['criterion_list']:
        crit.get_args(parser)

    m['metric_list']= load_wrappers_for_module_list(args.metrics, 'metrics')
    for metric in m['metric_list']:
        metric.get_args(parser)

    m['dataloader'] = load_module('dataloaders', 'dataloader').Dataloader(args.dataloader)
    m['dataloader'].get_args(parser)

    # Finally, `parser` is aware of all the possible parameters.
    # Go through the resolution order again to establish the values for all of them:
    # TODO make overriding verbose
    parser.set_defaults(**checkpoint_args)
    parser.set_defaults(**config_args)
    parser.set_defaults(**custom_args)
    args, default_args = parser.parse_args(), parser.parse_args([])

    # Dynamic defaults
    if not args.experiment_name:
        logger.info(f"`args.experiment_name` is missing, so setting it to `args.config_name` (\"{args.config_name}\")")
        args.experiment_name = args.config_name

    return args, default_args, m, checkpoint_object


def load_config_file(config_name):
    logger = logging.getLogger('utils.load_config_file')

    config_path = f'configs/{config_name}.yaml'

    logger.info(f"Using config {config_path}")
    with open(config_path, 'r') as stream:
        return yamlenv.load(stream)


def load_module(module_type, module_name):
    return importlib.import_module(f'{module_type}.{module_name}')


def load_wrappers_for_module_list(module_name_list: str, parent_module: str):
    """Import a comma-separated list of Python module names (e.g. "perceptual, adversarial,dice").
    For each of those modules, create the respective 'Wrapper' class and return those classes in
    a list."""
    module_names = module_name_list.split(',')
    module_names = [c.strip() for c in module_names if c.strip()]

    wrappers = []
    for module_name in module_names:
        module = importlib.import_module(f'{parent_module}.{module_name}')
        wrappers.append(module.Wrapper)

    return wrappers


class Meter:
    """
    Tracks average and last values of several metrics.
    """
    def __init__(self):
        super().__init__()
        self.sum = defaultdict(float)
        self.num_measurements = defaultdict(int)
        self.last_value = {}

    def add(self, name, value, num_measurements=1):
        """
        Add `num_measurements` measurements for metric `name`, given their average (`value`).
        To add just one measurement, call with `num_measurements = 1` (default).

        name:
            `str`
        value:
            convertible to `float`
        num_measurements:
            `int`
        """
        assert num_measurements >= 0
        if num_measurements == 0:
            return

        value = float(value)
        if value != value: # i.e. if value is NaN
            # add 0 in case dict keys don't exist yet
            self.sum[name] += 0
            self.num_measurements[name] += 0
        else:
            self.sum[name] += value * num_measurements
            self.num_measurements[name] += num_measurements
        self.last_value[name] = value

    def keys(self):
        return self.sum.keys()

    def get_average(self, name):
        return self.sum[name] / max(1, self.num_measurements[name])

    def get_last(self, name):
        return self.last_value[name]

    def get_num_measurements(self, name):
        return self.num_measurements[name]

    def __iadd__(self, other_meter):
        for name in other_meter.sum:
            self.add(name, other_meter.get_average(name), other_meter.get_num_measurements(name))
            self.last_value[name] = other_meter.last_value[name]
        return self


def save_model(training_module, optimizer_G, optimizer_D, args):
    logger = logging.getLogger('utils.save_model')

    if args.rank != 0:
        return

    if args.num_gpus > 1:
        training_module = training_module.module

    save_dict = {}
    if training_module.embedder is not None:
        save_dict['embedder'] = training_module.embedder.state_dict()
    if training_module.generator is not None:
        save_dict['generator'] = training_module.generator.state_dict()
    if training_module.discriminator is not None:
        save_dict['discriminator'] = training_module.discriminator.state_dict()
    if optimizer_G is not None:
        save_dict['optimizer_G'] = optimizer_G.state_dict()
    if optimizer_D is not None:
        save_dict['optimizer_D'] = optimizer_D.state_dict()
    if training_module.running_averages is not None:
        save_dict['running_averages'] = \
            {name: module.state_dict() for name, module in training_module.running_averages.items()}
    if args is not None:
        save_dict['args'] = args

    epoch_string = f'{args.iteration:08}'
    save_path = f'{args.experiment_dir}/checkpoints/model_{epoch_string}.pth'
    logger.info(f"Trying to save checkpoint at {save_path}")

    while os.path.exists(save_path): # ugly temporary solution, sorry
        epoch_string += '_0'
        save_path = f'{args.experiment_dir}/checkpoints/model_{epoch_string}.pth'
        logger.info(f"That path already exists, so augmenting it with '_0': {save_path}")

    try:
        logger.info(f"Finally saving checkpoint at {save_path}")
        torch.save(save_dict, save_path, pickle_protocol=-1)
        logger.info(f"Done saving checkpoint")
    except RuntimeError as err: # disk full?
        logger.error(f"Could not write to {save_path}: {err}; removing that file")
        try:
            os.remove(save_path)
        except:
            pass


def load_model_from_checkpoint(checkpoint_object, args=Namespace()):
    logger = logging.getLogger('utils.load_model_from_checkpoint')

    saved_args = checkpoint_object['args']
    saved_args_device_backup, saved_args.device = saved_args.device, 'cpu'

    # Determine
    # (1) if we will set the model to "fine-tuning mode" and
    # (2) if we have loaded a fine-tuned model
    finetune = 'finetune' in args and args.finetune
    already_finetuned = 'finetune' in saved_args and saved_args.finetune
    assert not (already_finetuned and 'finetune' in args and not finetune), \
         "NYI: using fine-tuned checkpoint for meta-learning"

    # TODO: move the below to `get_args_and_modules()`
    """
    # Load the command line arguments supplied when that model was trained
    # and combine them with the current arguments (if there are any)

    # Always prioritize values in checkpoint for these arguments
    ARGS_TO_IGNORE = \
        'iteration', 'num_labels'
    # Do not log the replacement of values for these arguments
    SILENT_ARGS = \
        ('device', 'fixed_val_ids', 'experiment_dir', 'local_rank', \
        'rank', 'world_size', 'num_workers', '__module__', '__dict__', '__weakref__', \
        'log_frequency_images', 'log_frequency_fixed_images', 'checkpoint_path', \
        'save_frequency', 'experiment_name', 'config_name')

    differing_args = []
    if args is not None:
        for arg_name, arg_value in vars(args).items():
            arg_value_saved = saved_args.__dict__.get(arg_name)
            if arg_value != arg_value_saved and arg_name not in ARGS_TO_IGNORE:
                if arg_name not in SILENT_ARGS:
                    logger.info(
                        f"Values for the config argument `{arg_name}` differ! Checkpoint has " \
                        f"`{arg_value_saved}`, replacing it with `{arg_value}`")

                differing_args.append(arg_name)
                saved_args.__dict__[arg_name] = args.__dict__[arg_name]
    """
    differing_args = []
    for arg_name, arg_value in vars(args).items():
        if arg_name in saved_args and arg_value != saved_args.__dict__.get(arg_name):
            differing_args.append(arg_name)

    # Load weights' running averages
    running_averages = checkpoint_object.get('running_averages', {})

    # Load embedder, generator, discriminator
    modules = {}
    for module_name in 'embedder', 'generator', 'discriminator':
        module_kind = getattr(args, module_name)
        logger.info(f"Loading {module_name} '{module_kind}'")

        module = load_module(f'{module_name}s', module_kind).Wrapper.get_net(args)

        module_old = load_module(f'{module_name}s', module_kind).Wrapper.get_net(saved_args)
        if already_finetuned:
            module_old.enable_finetuning()
        module_old.load_state_dict(checkpoint_object[module_name])

        # Change module structure to match the structure of the checkpointed module
        if finetune:
            module.enable_finetuning()
            if not already_finetuned:
                module_old.enable_finetuning()

        if module_name in differing_args:
            logger.warning(f"{module_name} has changed in config, so not loading weights")
        else:
            module.load_state_dict(module_old.state_dict())

        modules[module_name] = module

    # Load optimizer states, runner
    if 'inference' in args and args.inference:
        optimizer_G = optimizer_D = None
    else:
        optimizer_D = \
            load_module('discriminators', args.discriminator).Wrapper \
            .get_optimizer(modules['discriminator'], args)
        if 'discriminator' in differing_args or optimizer_D is None or finetune and not already_finetuned:
            logger.warning(f"Discriminator has changed in config (maybe due to finetuning), so not loading `optimizer_D`")
        else:
            optimizer_D.load_state_dict(checkpoint_object['optimizer_D'])

        logger.info(f"Loading runner {args.runner}")
        runner = load_module('runners', args.runner)
        optimizer_G = runner.get_optimizer(modules['embedder'], modules['generator'], args)
        if 'generator' in differing_args or 'embedder' in differing_args or finetune and not already_finetuned:
            logger.warning(f"Embedder or generator has changed in config, so not loading `optimizer_G`")
        else:
            optimizer_G.load_state_dict(checkpoint_object['optimizer_G'])

    saved_args.device = saved_args_device_backup

    return \
        modules['embedder'], modules['generator'], modules['discriminator'], \
        running_averages, saved_args, optimizer_G, optimizer_D


def torch_image_to_numpy(image_torch, inplace=False):
    """Convert PyTorch tensor to Numpy array.
    :param image_torch: PyTorch float CHW Tensor in range [0..1].
    :param inplace: modify the tensor in-place.
    :returns: Numpy uint8 HWC array in range [0..255]."""
    if not inplace:
        image_torch = image_torch.clone()
    return image_torch.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
