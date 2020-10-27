import time
import torch
from torch import nn
from tqdm import tqdm
import utils.radam
torch.optim.RAdam = utils.radam.RAdam

from utils.visualize import make_visual
from utils.utils import Meter
from utils import utils

import itertools
import copy

import logging # standard python logging
logger = logging.getLogger('runner')

def get_args(parser):
    parser.add('--iteration', type=int, default=0, help="Optional iteration number to start from")
    parser.add('--log_frequency_loss', type=int, default=1)
    parser.add('--log_frequency_images', type=int, default=100)
    parser.add('--log_frequency_fixed_images', type=int, default=2500)
    parser.add('--detailed_metrics', action='store_bool', default=True)
    # parser.add('--num_steps_dis_per_gen', default=2, type=int)
    parser.add('--num_visuals_per_img', default=2, type=int)
    parser.add('--fixed_val_ids', action='append', type=int, default=[50, 100, 200, 250, 300])
    parser.add('--batch_size_inference', default=5, type=int,
        help="Batch size for processing 'fixed_val_ids' during visualization. Different from 'batch_size', "
             "this number is for one GPU (visualization inference is currently done on 1 GPU anyway).")

    return parser


def get_optimizer(embedder, generator, args):
    model_parameters = list(generator.parameters())
    if 'finetune' not in args or not args.finetune: # TODO backward compatibility, remove `'finetune' not in args or`
        model_parameters += list(embedder.parameters())

    Optimizer = torch.optim.__dict__[args.optimizer]
    optimizer = Optimizer(model_parameters, lr=args.lr_gen, betas=(args.beta1, 0.999), eps=1e-5)
    return optimizer


class TrainingModule(torch.nn.Module):
    def __init__(self, embedder, generator, discriminator, criterion_list, metric_list, running_averages={}):
        """
            `embedder`, `generator`, `discriminator`: `nn.Module`s
            `criterion_list`, `metric_list`: a list of `nn.Module`s
            `running_averages`: `None` or a dict of {`str`: `nn.Module`}
                Optional initial states of weights' running averages (useful when resuming training).
                Can provide running averages just for some modules (e.g. for none or only for generator).
                If `None`, don't track running averages at all.
        """
        super().__init__()
        self.embedder = embedder
        self.generator = generator
        self.discriminator = discriminator
        self.criterion_list = nn.ModuleList(criterion_list)
        self.metric_list = nn.ModuleList(metric_list)

        self.compute_losses = True
        self.use_running_averages = False
        self.initialize_running_averages(running_averages)

    def initialize_running_averages(self, initial_values={}):
        """
            Set up weights' running averages for generator and discriminator.

            initial_values: `dict` of `nn.Module`, or `None`
                `None` means do not use running averages at all.

                Otherwise, `initial_values['embedder']` will be used as
                the initla value for embedder's running average. Same for generator.
                If `initial_values['embedder']` is missing, then embedder's running average
                will be initialized to embedder's current weights.
        """
        self.running_averages = {}

        if initial_values is not None:
            for name in 'embedder', 'generator':
                model = getattr(self, name)
                self.running_averages[name] = copy.deepcopy(model)
                try:
                    initial_value = initial_values[name]
                    self.running_averages[name].load_state_dict(initial_value)
                except KeyError:
                    logger.info(
                        f"No initial value of weights' running averages provided for {name}. Initializing by cloning")
                except:
                    logger.warning(
                        f"Parameters mismatch in {name} and the initial value of weights' "
                        f"running averages. Initializing by cloning")
                    self.running_averages[name].load_state_dict(model.state_dict())

        for module in self.running_averages.values():
            module.eval()
            module.requires_grad_(False)

    def update_running_average(self, alpha=0.999):
        with torch.no_grad():
            for model_name, model_running_avg in self.running_averages.items():
                model_current = getattr(self, model_name)

                for p_current, p_running_average in zip(model_current.parameters(), model_running_avg.parameters()):
                    p_running_average *= alpha
                    p_running_average += p_current * (1-alpha)

                for p_current, p_running_average in zip(model_current.buffers(), model_running_avg.buffers()):
                    p_running_average.copy_(p_current)

    def set_use_running_averages(self, use_running_averages=True):
        """
            Changes `training_module.use_running_averages` to the specified value.
            Can be used either as a context manager or as a separate method call.

            TODO: migrate to contextlib
        """
        class UseRunningAveragesContextManager:
            def __init__(self, training_module, use_running_averages):
                self.training_module = training_module
                self.old_value = self.training_module.use_running_averages
                self.training_module.use_running_averages = use_running_averages
            
            def __enter__(self):
                pass

            def __exit__(self, *args):
                self.training_module.use_running_averages = self.old_value

        return UseRunningAveragesContextManager(self, use_running_averages)

    def set_compute_losses(self, compute_losses=True):
        """
            Changes `training_module.compute_losses` to the specified value.
            Can be used either as a context manager or as a separate method call.

            TODO: migrate to contextlib
        """
        class ComputeLossesContextManager:
            def __init__(self, training_module, compute_losses):
                self.training_module = training_module
                self.old_value = self.training_module.compute_losses
                self.training_module.compute_losses = compute_losses
            
            def __enter__(self):
                pass

            def __exit__(self, *args):
                self.training_module.compute_losses = self.old_value

        return ComputeLossesContextManager(self, compute_losses)

    def forward(self, data_dict, target_dict):
        if self.running_averages and self.use_running_averages:
            embedder = self.running_averages['embedder']
            generator = self.running_averages['generator']
        else:
            embedder = self.embedder
            generator = self.generator

        # First, let the new `data_dict` (the return value) hold only inputs.
        # As we run modules, `data_dict` will gradually get filled with outputs too.
        data_dict = copy.copy(data_dict)

        embedder(data_dict)
        generator(data_dict)

        # Now add target data to `data_dict` too, to run discriminator and calculate losses if needed
        data_dict.update(target_dict)
        if self.compute_losses:
            self.discriminator(data_dict)
        
        losses_G_dict = {}
        losses_D_dict = {}

        for criterion in self.criterion_list:
            try:
                crit_out = criterion(data_dict)
            except:
                # couldn't compute this loss; if validating, skip it
                if self.compute_losses:
                    raise
                else:
                    continue

            if isinstance(crit_out, tuple):
                if len(crit_out) != 2:
                    raise TypeError(
                        f'Unexpected number of outputs in criterion {type(criterion)}: '
                        f'expected 2, got {len(crit_out)}')
                crit_out_G, crit_out_D = crit_out
                losses_G_dict.update(crit_out_G)
                losses_D_dict.update(crit_out_D)
            elif isinstance(crit_out, dict):
                losses_G_dict.update(crit_out)
            else:
                raise TypeError(
                    f'Unexpected type of {type(criterion)} output: '
                    f'expected dict or tuple of two dicts, got {type(crit_out)}')

        return data_dict, losses_G_dict, losses_D_dict

    def compute_metrics(self, data_dict):
        metrics_meter = Meter()
        for metric in self.metric_list:
            metric_out, num_errors = metric(data_dict)
            for metric_output_name, metric_value in metric_out.items():
                metrics_meter.add(metric_output_name, metric_value, num_errors[metric_output_name])

        return metrics_meter

def run_epoch(dataloader, training_module, optimizer_G, optimizer_D, epoch, args,
              phase,
              writer=None,
              saver=None):
    meter = Meter()

    if phase == 'train':
        optimizer_G.zero_grad()
        if optimizer_D:
            optimizer_D.zero_grad()

    end = time.time()
    for it, (data_dict, target_dict) in enumerate(dataloader):
        meter.add('Data_time', time.time() - end)

        utils.dict_to_device(data_dict  , args.device)
        utils.dict_to_device(target_dict, args.device)

        all_data_dict, losses_G_dict, losses_D_dict = training_module(data_dict, target_dict)

        # The isinstance() check is a workaround. We have some metrics that are implemented as losses
        # (currently gaze angle metric), and those are not differentiable, and occasionally take NaN values.
        # We don't need to include them in the cumulative loss_{G,D} values, as those are used for backprop only.
        loss_G = sum(v for v in losses_G_dict.values() if isinstance(v, torch.Tensor))
        loss_D = sum(v for v in losses_D_dict.values() if isinstance(v, torch.Tensor))

        if phase == 'train':
            optimizer_G.zero_grad()
            loss_G.backward(retain_graph=True)
            if args.num_gpus > 1 and args.num_gpus <= 8:
                training_module.reducer.reduce()

            optimizer_G.step()
            
            if losses_D_dict:
                optimizer_D.zero_grad()
                loss_D.backward()
                if args.num_gpus > 1 and args.num_gpus <= 8:
                    training_module.reducer.reduce()

                optimizer_D.step()

        if phase == 'val' and saver is not None:
            saver.save(epoch=epoch, data=all_data_dict)

        training_module.update_running_average(0.972 if args.finetune else 0.999)

        # Log
        if args.detailed_metrics:
            for loss_name, loss_ in itertools.chain(losses_G_dict.items(), losses_D_dict.items()):
                meter.add(f'Loss_{loss_name}', float(loss_))

        del loss_G, loss_D, losses_G_dict, losses_D_dict

        def try_other_driving_images(data_dict, suffix, same_identity=False, deterministic=False):
            """
            For each sample in the given `data_dict`, pick a different driving image
            ('pose_input_rgbs'), run the model again with those drivers and save its new outputs
            ('fake_rgbs', 'fake_segm' etc.) -- as well as new inputs (e.g. 'pose_input_rgbs') --
            to `data_dict` with a given `suffix` (e.g. 'pose_input_rgbs_cross_driving' and
            'fake_rgbs_cross_driving' if `suffix` is '_cross_driving').

            data_dict:
                `dict`
                As returned by dataloaders.
            suffix:
                `str`
                See above.
            same_identity:
                `bool`
                If `True`, pick new drivers from other (if possible) videos of the same person.
                Else, pick them from videos of other people.
            deterministic:
                `bool`
                Whether to choose fixed (`True`) or random (`False`) drivers.
            """
            given_samples_labels = data_dict['label'].tolist()

            # Why double `.dataset`? Because dataloader links to a `torch.utils.data.Subset`, which
            # in turn links to our full VoxCeleb dataset
            other_samples_indices = \
                [dataloader.dataset.dataset.get_other_sample_by_label(l, same_identity=same_identity,\
                    deterministic=deterministic) for l in given_samples_labels]

            other_samples = [dataloader.dataset.dataset[i][0] for i in other_samples_indices]
            other_samples = dataloader.collate_fn(other_samples)

            # First, backup original inputs for visualization
            keys_to_backup = \
                'pose_input_rgbs', 'target_rgbs', '3dmm_pose', \
                'fake_rgbs', 'real_segm', 'fake_segm', 'dec_stickmen', 'dec_keypoints'
            backup = {key: data_dict[key] for key in keys_to_backup if key in data_dict}
            # Then replace that data with new inputs
            for key in keys_to_backup:
                if key in other_samples:
                    data_dict[key] = other_samples[key].to(args.device)

            updated_data_dict, _, _ = training_module(data_dict, {})    
            data_dict.update(updated_data_dict)

            # Finally, save new inputs and outputs by a new key, and restore backup
            for key in backup:
                if key in data_dict:
                    data_dict[key + suffix] = data_dict[key]
                    data_dict[key] = backup[key]


        if writer is not None and phase == 'train':
            if args.iteration % args.log_frequency_loss == 0:
                for metric in meter.keys():
                    writer.add_scalar(f'Metrics/{phase}/{metric}', meter.get_last(metric), args.iteration)

            if args.iteration % args.log_frequency_images == 0:
                # Visualize how a person drives self but from other video

                # Re-evaluate embedding because embedder can behave differently in .train() and .eval()
                training_module.train(not args.set_eval_mode_in_test)
                with torch.no_grad():
                    with training_module.set_use_running_averages(), training_module.set_compute_losses(False):
                        all_data_dict, _, _ = training_module(data_dict, {'label': target_dict['label']})
                        if not args.finetune:
                            try_other_driving_images(all_data_dict, suffix='_other_video', same_identity=True)
                            try_other_driving_images(all_data_dict, suffix='_other_person', same_identity=False)
                training_module.train(not args.set_eval_mode_in_train)

                try:
                    del all_data_dict['dec_stickmen']
                except KeyError:
                    pass
                logging_images, captions = make_visual(all_data_dict, n_samples=args.num_visuals_per_img)
                writer.add_image(f'Images/{phase}/visual', logging_images, captions, args.iteration)
                
            if args.iteration % args.log_frequency_fixed_images == 0 and args.fixed_val_ids:
                # Make data loading functions deterministic to make sure same images are sampled from a directory
                was_deterministic = dataloader.dataset.dataset.loader.deterministic
                dataloader.dataset.dataset.loader.deterministic = True

                metrics_meter = Meter()

                # Also, make augmentations always same for each sample in batch
                with dataloader.dataset.dataset.deterministic_(666):
                    # Iterate over `args.fixed_data_dict` in batches
                    for first_sample_idx in range(0, len(args.fixed_val_ids), args.batch_size_inference):
                        batch_sample_ids = args.fixed_val_ids[first_sample_idx:first_sample_idx+args.batch_size_inference]
                        fixed_data_dict = [dataloader.dataset.dataset[i] for i in batch_sample_ids]
                        fixed_data_dict, fixed_target_dict = dataloader.collate_fn(fixed_data_dict)
                        fixed_data_dict.update(fixed_target_dict)
                        utils.dict_to_device(fixed_data_dict, args.device)

                        training_module.train(not args.set_eval_mode_in_test)
                        with torch.no_grad():
                            with training_module.set_use_running_averages(), training_module.set_compute_losses(False):
                                fixed_all_data_dict, _, _ = training_module(fixed_data_dict, {})
                                if not args.finetune:
                                    try_other_driving_images(fixed_all_data_dict, suffix='_other_video', same_identity=True, deterministic=True)
                                    try_other_driving_images(fixed_all_data_dict, suffix='_other_person', same_identity=False, deterministic=True)
                        training_module.train(not args.set_eval_mode_in_train)

                        try:
                            del fixed_all_data_dict['dec_stickmen']
                        except KeyError:
                            pass

                        # Visualize the first batch in TensorBoard/WandB
                        if first_sample_idx == 0:
                            logging_images, captions = make_visual(fixed_all_data_dict, n_samples=len(batch_sample_ids))
                            writer.add_image(f'Fixed_images/{phase}/visual', logging_images, captions, args.iteration)

                        with torch.no_grad():
                            metrics_meter += training_module.compute_metrics(fixed_all_data_dict)

                for metric_name in metrics_meter.keys():
                    writer.add_scalar(
                        f'Fixed_metrics/{phase}/{metric_name}', metrics_meter.get_average(metric_name), args.iteration)

                dataloader.dataset.dataset.loader.deterministic = was_deterministic

            if phase == 'train':
                args.iteration += 1

        # Measure elapsed time
        meter.add('Batch_time', time.time() - end)
        end = time.time()

    if writer is not None and phase == 'val':
        for metric in meter.keys():
            writer.add_scalar(f'Metrics/{phase}/{metric}', meter.get_average(metric), args.iteration)
        logging_images, captions = make_visual(all_data_dict, n_samples=args.num_visuals_per_img * 3)
        writer.add_image(f'Images/{phase}/visual', logging_images, captions, args.iteration)

    logger.info(f"Epoch {epoch} {phase.capitalize()} finished")
