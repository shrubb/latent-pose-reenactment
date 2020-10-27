import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
from utils.utils import load_module

import logging
logger = logging.getLogger('dataloaders.dataloader')

class Dataloader:
    def __init__(self, dataset_name):
        self.dataset = self.find_definition(dataset_name)

    def find_definition(self, dataset_name):
        m = load_module('dataloaders', dataset_name)
        return m.__dict__['Dataset']

    def get_args(self, parser):
        parser.add('--num_workers', type=int, default=4, help='Number of data loading workers.')
        parser.add('--prefetch_size', type=int, default=16, help='Prefetch queue size')
        parser.add('--batch_size', type=int, default=64, help='Batch size')

        return self.dataset.get_args(parser)

    def get_dataloader(self, args, part, phase):
        if hasattr(self.dataset, 'get_dataloader'):
            return self.dataset.get_dataloader(args, part)
        else:
            dataset = self.dataset.get_dataset(args, part)
            # Get a split for this process in distributed training
            assert len(dataset) % args.world_size == 0, \
                "`dataset.get_dataset()` was expected to return a dataset equally divisible by `args.world_size`"
            dataset = torch.utils.data.Subset(dataset, range(args.rank, len(dataset), args.world_size))

            logger.info(f"This process will receive a dataset with {len(dataset)} samples")

            if len(dataset) < args.batch_size: # can happen at fine-tuning
                logger.warning(
                    f"Dataset length is smaller than batch size ({len(dataset)} < {args.batch_size})" \
                    f", reducing the latter to {len(dataset)}")
                args.batch_size = len(dataset)

            return DataLoaderWithPrefetch(
                dataset,
                prefetch_size=args.prefetch_size,
                batch_size=args.batch_size // args.num_gpus,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True if phase == 'train' else False,
                shuffle=True if part == 'train' else False)


class DataLoaderWithPrefetch(DataLoader):
    def __init__(self, *args, prefetch_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefetch_size = prefetch_size if prefetch_size is not None else 2 * kwargs.get('num_workers', 0)

    def __iter__(self):
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIterWithPrefetch(self)


class _MultiProcessingDataLoaderIterWithPrefetch(_MultiProcessingDataLoaderIter):
    def __init__(self, loader):
        self.prefetch_size = loader.prefetch_size

        super().__init__(loader)

        # Prefetch more items than the default 2 * self._num_workers
        assert self.prefetch_size >= 2 * self._num_workers
        for _ in range(loader.prefetch_size - 2 * self._num_workers):
            self._try_put_index()

    def _try_put_index(self):
        assert self._tasks_outstanding < self.prefetch_size
        try:
            index = self._next_index()
        except StopIteration:
            return
        for _ in range(self._num_workers):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            # not found (i.e., didn't break)
            return

        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1
