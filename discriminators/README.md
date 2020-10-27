# Discriminators

To create a new discriminator create a `.py` file in this directory.
This file should specify a class `Wrapper` with three static methods:

* `get_params()` -- declares arguments for your embedder
    * takes `utils.argparse_utils.MyArgumentParser` instance as an input
    * appends arguments to the parser, that may/should be specified in the config file 
* `get_net()` -- returns an instance of Discriminator class inherited from `torch.nn.Module`
* `get_optimizer()` -- returns a `torch.optim.Optimizer`. Takes the discriminator `torch.nn.Module` and the arguments via `utils.argparse_utils.MyArgumentParser`.

Discriminators' `forward()` methods should take a dictionary with all data `data_dict` (initially provided by a [dataloader](../dataloaders)) augmented with outputs of [generator](../generators) and [embedder](../embedders). Discriminator adds to `data_dict` any data needed to calculate adversarial and other loss values for both generator loss and discriminator loss.

In addition, discriminators have to define `enable_finetuning(self, data_dict=None)` (see the [documentation for generators](../generators)).