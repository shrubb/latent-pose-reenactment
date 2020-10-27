import os, sys
os.environ['OMP_NUM_THREADS'] = '1'

import torch
from torch.optim import Adam
from torch import nn
from pathlib import Path

from utils import utils
from utils.argparse_utils import MyArgumentParser
from utils.utils import setup, get_args_and_modules, save_model, load_model_from_checkpoint
from utils.tensorboard_logging import setup_logging
from utils.visualize import Saver

import logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="PID %(process)d - %(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger('train.py')

parser = MyArgumentParser(conflict_handler='resolve') # TODO: allow_abbrev=False
parser.add = parser.add_argument

parser.add('--config_name', type=str, default="")

parser.add('--generator', type=str, default="", help='')
parser.add('--embedder', type=str, default="", help='')
parser.add('--discriminator', type=str, default="", help='')
parser.add('--criterions', type=str, default="", help='')
parser.add('--metrics', type=str, default="", help='')
parser.add('--dataloader', type=str, default="", help='')
parser.add('--runner', type=str, default="", help='')

parser.add('--args-to-ignore', type=str,
           default="checkpoint,splits_dir,experiments_dir,extension,"
                   "experiment_name,rank,local_rank,world_size")
parser.add('--experiments_dir', type=Path, default="data/experiments", help='')
parser.add('--experiment_name', type=str, default="", help='')
parser.add('--train_split_path', default="data/splits/train.csv", type=Path,
    help="Enumerates identities from the dataset to be used in training. Resolution order: " \
         "if '`--data_root`/`--img_dir`/`--train_split_path`' is a valid directory, use that; " \
         "if '`--train_split_path`' points to an existing file, use it as a CSV identity list; " \
         "else, use sorted list of directories '`--data_root`/`--img_dir`/*'.")
parser.add('--val_split_path', default="data/splits/val.csv", type=Path,
    help="See `--train_split_path`.")

# directory with vgg weights for perceptual losses
parser.add('--vgg_weights_dir', default="criterions/common/", type=str)

# Training process
parser.add('--num_epochs', type=int, default=10**9)
parser.add('--set_eval_mode_in_train', action='store_bool', default=False)
parser.add('--set_eval_mode_in_test', action='store_bool', default=True)
parser.add('--save_frequency', type=int, default=1,
    help="Save checkpoint every X epochs. If 0, save only at the end of training")
parser.add('--logging', action='store_bool', default=True)
parser.add('--skip_eval', action='store_bool', default=True)
parser.add('--profile_flops', action='store_bool', default=False)
parser.add('--weights_running_average', action='store_bool', default=True)
parser.add('--finetune', action='store_bool', default=False)
parser.add('--inference', action='store_bool', default=False)

# Model
parser.add('--in_channels', type=int, default=3)
parser.add('--out_channels', type=int, default=3)
parser.add('--num_channels', type=int, default=64)
parser.add('--max_num_channels', type=int, default=512)
parser.add('--embed_channels', type=int, default=512)
parser.add('--pose_embedding_size', type=int, default=136)
parser.add('--image_size', type=int, default=256)

# Optimizer
parser.add('--optimizer', default='Adam', type=str, choices=['Adam', 'RAdam'])
parser.add('--lr_gen', default=5e-5, type=float)
parser.add('--beta1', default=0.0, type=float, help='beta1 for Adam')

# Hardware
parser.add('--device', type=str, default='cuda')
parser.add('--num_gpus', type=int, default=1, help='requires apex if > 1, requires horovod if > 8')
parser.add('--hvd_fp16_allreduce', action='store_true')
parser.add('--hvd_batches_per_allreduce', default=1, help='number of batches processed locally before allreduce')
parser.add('--rank', type=int, default=0, help='global rank, DO NOT SET')
parser.add('--local_rank', type=int, default=0, help='"rank" within a machine, DO NOT SET')
parser.add('--world_size', type=int, default=1, help='number of devices, DO NOT SET')

# Misc
parser.add('--random_seed', type=int, default=123, help='')
parser.add('--checkpoint_path', type=str, default='')
parser.add('--saver', type=str, default='')

args, default_args, m, checkpoint_object = get_args_and_modules(parser, use_checkpoint_args=True)

# Set random seed, number of threads etc.
setup(args)

# In case of distributed training we first initialize rank and world_size
if args.num_gpus == 1:
    args.rank = args.local_rank = 0
    args.world_size = 1
elif args.num_gpus > 1 and args.num_gpus <= 8:
    # use distributed data parallel
    # `args.local_rank` is the automatic command line argument, input by `python -m torch.distributed.launch ...`;
    # `args.rank` is the actual rank value we rely on
    args.rank = args.local_rank
    args.world_size = args.num_gpus
    torch.cuda.set_device(args.local_rank)
    args.device = f'cuda:{args.local_rank}'
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
elif args.num_gpus > 8:
    # use horovod
    import horovod.torch as hvd
    hvd.init()
    # set options
    args.rank = hvd.rank()
    args.world_size = hvd.size()

logger.info(f"Initialized the process group, my rank is {args.rank}")

if args.finetune and args.num_gpus > 1:
    if args.local_rank == 0:
        logger.warning("Sorry, multi-GPU fine-tuning is NYI, setting `--num_gpus=1`")
        args.num_gpus = 1
    else:
        logger.warning("Sorry, multi-GPU fine-tuning is NYI, shutting down all processes but one")
        exit()

logger.info(f"Loading dataloader '{args.dataloader}'")
dataloader_train = m['dataloader'].get_dataloader(args, part='train', phase='train')
if not args.skip_eval:
    if args.num_gpus > 1:
        raise NotImplementedError("Multi-GPU validation not implemented")
    dataloader_val = m['dataloader'].get_dataloader(args, part='val', phase='val')

runner = m['runner']

if args.checkpoint_path != "":
    if checkpoint_object is not None:
        logger.info(f"Starting from checkpoint {args.checkpoint_path}")
        embedder, generator, discriminator, \
        running_averages, saved_args, optimizer_G, optimizer_D = \
            load_model_from_checkpoint(checkpoint_object, args)

        logger.info(f"Starting from iteration #{args.iteration}")
    else:
        raise FileNotFoundError(f"Checkpoint `{args.checkpoint_path}` not found")
else:
    if args.finetune:
        logger.error("`--finetune` is set, but `--checkpoint_path` isn't. This has to be a mistake.")

    discriminator = m['discriminator'].get_net(args)
    generator = m['generator'].get_net(args)
    embedder = m['embedder'].get_net(args)
    running_averages = {}

    optimizer_G = runner.get_optimizer(embedder, generator, args)
    optimizer_D = m['discriminator'].get_optimizer(discriminator, args)

criterion_list = [crit.get_net(args) for crit in m['criterion_list']]

if not args.weights_running_average:
    running_averages = None

writer = None
if args.logging and args.rank == 0:
    args.experiment_dir, writer = setup_logging(
        args, default_args, args.args_to_ignore.split(','))
    args.experiment_dir = Path(args.experiment_dir)
    metric_list = [metric.get_net(args) for metric in m['metric_list']]
else:
    metric_list = []

training_module = runner.TrainingModule(embedder, generator, discriminator, criterion_list, metric_list, running_averages)

# If someone tries to terminate the program, let us save the weights first
model_already_saved = False
if args.rank == 0:
    import signal, sys, os
    parent_pid = os.getpid()
    def save_last_model_and_exit(_1, _2):
        global model_already_saved
        if model_already_saved:
            return
        model_already_saved = True
        if os.getpid() == parent_pid: # otherwise, dataloader workers will try to save the model too!
            logger.info("Interrupted, saving the current model")
            save_model(training_module, optimizer_G, optimizer_D, args)
            # protect from Tensorboard's "Unable to get first event timestamp
            # for run `...`: No event timestamp could be found"
            if writer is not None:
                writer.close()
            sys.exit()
    signal.signal(signal.SIGINT , save_last_model_and_exit)
    signal.signal(signal.SIGTERM, save_last_model_and_exit)

if args.device.startswith('cuda') and torch.cuda.device_count() > 1:
    if args.num_gpus > 1 and args.num_gpus <= 8:
        from apex.parallel import Reducer
        training_module.reducer = Reducer(training_module)
        training_module.__dict__['module'] = training_module # do not register self as a nested module
    elif args.num_gpus > 8:
        optimizer_G = hvd.DistributedOptimizer(optimizer_G,
                                               named_parameters=runner.get_named_parameters(embedder, generator, args),
                                               compression=hvd.Compression.fp16 if args.hvd_fp16_allreduce else hvd.Compression.none,
                                               backward_passes_per_step=args.hvd_batches_per_allreduce)
        optimizer_D = hvd.DistributedOptimizer(optimizer_G,
                                               named_parameters=m['discriminator'].get_named_parameters(discriminator, args),
                                               compression=Compression.fp16 if args.hvd_fp16_allreduce else hvd.Compression.none,
                                               backward_passes_per_step=args.hvd_batches_per_allreduce)
        hvd.broadcast_optimizer_state(optimizer_G, root_rank=0)
        hvd.broadcast_optimizer_state(optimizer_D, root_rank=0)

# Optional results saver
saver = None
if args.saver and args.rank == 0:
    saver = Saver(save_dir=f'{args.experiment_dir}/validation_results/', save_fn=args.saver)

if args.finetune:
    # A dirty hack (sorry) for reproducing X2Face within our pipeline
    if args.generator == 'X2Face':
        MAX_IDENTITY_IMAGES = 8
        identity_images = []
        for data_dict, _ in dataloader_train:
            identity_images.append(data_dict['pose_input_rgbs'][:, 0]) # B x C x H x W
            total_identity_images = sum(map(len, identity_images))
            if total_identity_images >= MAX_IDENTITY_IMAGES:
                break

        total_identity_images = min(MAX_IDENTITY_IMAGES, total_identity_images)

        logger.info(f"Saving X2Face model with {total_identity_images} identity images")
        args.X2Face_num_identity_images = total_identity_images
        data_dict = {'enc_rgbs': torch.cat(identity_images)[:total_identity_images][None]}
        training_module.generator.enable_finetuning(data_dict)

        print(training_module.generator.identity_images.shape)
        save_model(training_module, optimizer_G, optimizer_D, args)
        exit()

    logger.info(f"For fine-tuning, computing an averaged identity embedding from {len(dataloader_train.dataset)} frames")

    training_module.eval()
    identity_embeddings = []

    with torch.no_grad():
        # Precompute identity embedding $\hat{e}_{NEW}$
        for data_dict, _ in dataloader_train:
            try:
                embedder = training_module.running_averages['embedder']
            except:
                logger.warning(f"Couldn't get embedder's running average, computing the embedding with the original embedder")
                embedder = training_module.embedder

            utils.dict_to_device(data_dict, args.device)
            embedder.get_identity_embedding(data_dict)
            identity_embeddings.append(data_dict['embeds_elemwise'].view(-1, args.embed_channels))

        identity_embedding = torch.cat(identity_embeddings).mean(0)
        del identity_embeddings

    # Initialize person-specific generator parameters $\psi'$ and flag them as trainable
    data_dict = {'embeds': identity_embedding[None]}
    training_module.generator.enable_finetuning(data_dict)
    # Put the embedding $\hat{e}_{NEW}$ into discriminator's matrix W
    training_module.discriminator.enable_finetuning(data_dict)

    if args.weights_running_average:
        # Do the same for running averages
        if 'generator' in training_module.running_averages:
            training_module.running_averages['generator'].enable_finetuning(data_dict)
        if 'discriminator' in training_module.running_averages:
            training_module.running_averages['discriminator'].enable_finetuning(data_dict)
    else:
        # Remove running averages
        training_module.initialize_running_averages(None)

    # Re-initialize optimizers
    optimizer_G = runner.get_optimizer(training_module.embedder, training_module.generator, args)
    optimizer_D = m['discriminator'].get_optimizer(discriminator, args)

logger.info(f"Entering training loop")

# Main loop
for epoch in range(0, args.num_epochs):
    # ===================
    #       Train
    # ===================
    training_module.train(not args.set_eval_mode_in_train)
    torch.set_grad_enabled(True)
    runner.run_epoch(dataloader_train, training_module, optimizer_G, optimizer_D,
                     epoch, args, phase='train', writer=writer, saver=saver)

    if not args.skip_eval:
        raise NotImplementedError("NYI: validation")
        # ===================
        #       Validate
        # ===================
        training_module.train(not args.set_eval_mode_in_test)
        torch.set_grad_enabled(False)
        with training_module.set_use_running_averages(), training_module.set_compute_losses(False):
            runner.run_epoch(dataloader_val, training_module, None, None,
                             epoch, args, phase='val', writer=writer, saver=saver)

    if args.rank == 0:
        will_save_checkpoint = epoch == args.num_epochs-1
        if args.save_frequency != 0:
            will_save_checkpoint |= epoch % args.save_frequency == 0

        if will_save_checkpoint:
            save_model(training_module, optimizer_G, optimizer_D, args)
