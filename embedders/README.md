# Embedders

To create a new embedder, create a `.py` file in this directory.
This file should specify a class `Wrapper` with two static methods:
* `get_params()` -- declares arguments for your embedder
    * takes `utils.argparse_utils.MyArgumentParser` instance as an input
    * appends arguments to the parser, that may/should be specified in the config file 
* `get_net()` -- returns an instance of Embedder class inherited from `torch.nn.Module`

An embedder should implement the following methods:
* `get_identity_embedding(self, data_dict)`: takes a `data_dict` provided by a [dataloader](../dataloaders), and adds any output elements to it after computing those from tensors related to `data_dict['enc_rgbs']`. In `vanilla` embedder, this adds `data_dict['embeds']` (aggregated identity embedding) and `data_dict['embeds_elemwise']` (non-aggregated identity embeddings for each image in `data_dict['enc_rgbs']`).
* `get_pose_embedding(self, data_dict)`: same as `get_identity_embedding()`, but computes *pose* information from tensors related to `data_dict['pose_input_rgbs']`. In `vanilla` embedder, this does nothing, because all pose information (`data_dict['dec_stickmen']`) is obtained from the dataloader and is consumed by the generator directly. In latent pose embedders, it takes `data_dict['pose_input_rgbs']` for example and computes `data_dict['pose_embedding']`.
* `enable_finetuning(self, data_dict=None)`: see the [documentation for generators](../generators).
