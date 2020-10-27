# Criterions (losses)

**Yes, we know it should be "criteria". Legacy stuff, sorry, too hard to refactor 😫**

In your config file you can specify several losses, for example:

```yaml
criterions: adversarial, perceptual, featmat
```

Added up, they will constitute the final training loss.

### Wrapper

Losses reside in their `.py` files. In each of them, we need to specify a class named `Wrapper` with two `@staticmethod`s:

* `get_args(parser)`
    * takes a `utils.argparse_utils.MyArgumentParser` as input;
    * appends arguments to the parser, that may/should be specified in the config file.
* `get_net(args)` returns a Criterion instance (inherited from `torch.nn.Module`), taking into account the passed arguments.

That instance must implement `forward()` method like so:

### `Criterion.forward(data_dict)`

This method takes a `dict` that is a merge of

* `data_dict` as returned by dataloader
* and any extra values added to it during the network's forward pass, and
* `target_dict` as returned by dataloader.

Depending on the criterion's purpose, it may output one or two dicts.

**One dict output:** `loss_G_dict` containing loss values for generator+embedder.

**Two dicts output:** `(loss_G_dict, loss_D_dict)`, *in that order*.

*`loss_G_dict` containing loss values for generator+embedder part;
*`loss_D_dict` containing loss values for discriminator part.
