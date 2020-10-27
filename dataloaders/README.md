# Dataloaders

* `dataloader.py` hosts the `torch.utils.data.Dataloader` class to wrap custom datasets.
It has several own parameters (`num_workers`, `batch_size`) and shouldn't be changed.

To make a custom dataset, create a `.py` file in this directory. This file should contain a class named `Dataset` with two `@staticmethod`s:

* `get_args(parser)`
    * takes `utils.argparse_utils.MyArgumentParser` instance as an input;
    * appends arguments to the parser, that may/should be specified in the `--config` file.
* `get_dataset(args)`
    * returns a `torch.utils.data.Dataset`, where each 'sample' (as returned by `__getitem__`) is two `dict`s:
        * `data_dict` with tensors/numbers/strings used somewhere in the forward pass;
        * `target_dict` with everything else (e.g. tensors that are only used to calculate loss values).
