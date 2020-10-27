# Generators

To create a new generator, create a `.py` file in this directory.
This file should specify a class `Wrapper` with two static methods:
* `get_params()` -- declares arguments for your embedder
    * takes `utils.argparse_utils.MyArgumentParser` instance as an input
    * appends arguments to the parser, that may/should be specified in the config file 
* `get_net()` -- returns an instance of Generator class inherited from `torch.nn.Module`

A generator should implement the following methods:
* `forward(self, data_dict)`: takes a `data_dict` provided by a [dataloader](../dataloaders) and adds any other elements to it 
(e.g. `vanilla` generator adds `'fake_rgbs'`).
* `enable_finetuning(self, data_dict=None)`: sets the instance into the "finetuning" mode. This may impact future calls of functions like `forward()` or `parameters()`. In `vanilla` generator, this will set `self.finetuning = True`, compute AdaIN parameters, mark them as trainable (i.e. make them `torch.nn.Parameter`s), and will prevent `forward()` from using `data_dict['embeds']` and thus from computing AdaIN parameters.
