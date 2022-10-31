# Neural Head Reenactment with Latent Pose Descriptors

![](https://user-images.githubusercontent.com/9570420/94962966-0a8bb900-0500-11eb-90ee-3315368019b8.png)

Burkov, E., Pasechnik, I., Grigorev, A., & Lempitsky V. (2020, June). **Neural Head Reenactment with Latent Pose Descriptors**. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

See the [project page](https://saic-violet.github.io/latent-pose-reenactment/) for an overview.

## Prerequisites

For fine-tuning a pre-trained model, you'll need an NVIDIA GPU, preferably with 8+ GB VRAM. To train from scratch, we recommend a total of 40+ GB VRAM.

Set up your environment as described [here](INSTALL.md).

## Running the pretrained model

* Collect images of the person to reenact.
* Run [`utils/preprocess_dataset.sh`](utils/preprocess_dataset.sh) to preprocess them. Read inside for instructions.
* Download the meta-model [checkpoint](https://drive.google.com/file/d/14-FYaz6YhTX5M_P3-rm2ITcxGljmWl-F/view?usp=share_link).
* Run the below to fine-tune the meta-model to your person, first setting the top variables. If you want, also launch a TensorBoard at `"$OUTPUT_PATH"` to view progress, preferably with the [`--samples_per_plugin "scalars=1000,images=100"`](https://stackoverflow.com/questions/57669234/how-to-display-more-than-10-images-in-tensorboard) option; mainly check the "images" tab to find out at which iteration the identity gap becomes small enough.

```bash
# in this example, your images should be "$DATASET_ROOT/images-cropped/$IDENTITY_NAME/*.jpg"
DATASET_ROOT="/where/is/your/data"
IDENTITY_NAME="identity/name"
MAX_BATCH_SIZE=8             # pick the largest possible, start with 8 and decrease until it fits in VRAM
CHECKPOINT_PATH="/where/is/checkpoint.pth"
OUTPUT_PATH="outputs/"       # a directory for outputs, will be created
RUN_NAME="tony_hawk_take_1"  # give your run a name if you want

# Important. See the note below
TARGET_NUM_ITERATIONS=230

# Don't change these
NUM_IMAGES=`ls -1 "$DATASET_ROOT/images-cropped/$IDENTITY_NAME" | wc -l`
BATCH_SIZE=$((NUM_IMAGES<MAX_BATCH_SIZE ? NUM_IMAGES : MAX_BATCH_SIZE))
ITERATIONS_IN_EPOCH=$(( NUM_IMAGES / BATCH_SIZE ))

mkdir -p $OUTPUT_PATH

python3 train.py \
    --config finetuning-base                 \
    --checkpoint_path "$CHECKPOINT_PATH"     \
    --data_root "$DATASET_ROOT"              \
    --train_split_path "$IDENTITY_NAME"      \
    --batch_size $BATCH_SIZE                 \
    --num_epochs $(( (TARGET_NUM_ITERATIONS + ITERATIONS_IN_EPOCH - 1) / ITERATIONS_IN_EPOCH )) \
    --experiments_dir "$OUTPUT_PATH"         \
    --experiment_name "$RUN_NAME"
```

**Note**. `TARGET_NUM_ITERATIONS` is important, make sure to tune it. Pick too low, underfit and get an identity gap; pick too high, overfit and get poor mimics. I suggest that you start with **125 when `NUM_IMAGES=1`** and increase with more images, say, to **230 when `NUM_IMAGES>30`**. But your concrete case may be different. If you have a lot of disk space, pass a flag to save checkpoints every so often (e.g. `--save_frequency 4` will save a checkpoint every `4 * NUM_IMAGES` iterations), then drive (see below how) each of them and thus find the iteration where the best tradeoff happens for your avatar.

* Take your driving video and crop it with `python3 utils/crop_as_in_dataset.py`. Run with `--help` to learn how. Or, equivalently, just reuse [`utils/preprocess_dataset.sh`](utils/preprocess_dataset.sh) with `COMPUTE_SEGMENTATION=false`.
* Organize the cropped images from the previous step as `"<data_root>/images-cropped/<images_path>/*.jpg"`.
* Use them to drive your fine-tuned model (the checkpoint is at `"$OUTPUT_PATH/$RUN_NAME/checkpoints"`) with `python3 drive.py`. Run with `--help` to learn how.

## Training (meta-learning) your own model

You'll need a training configuration (aka config) file. Start with `"configs/default.yaml"` or just edit that. These files specify various training options which you can find in code as `argparse` parameters. Any of these options can be specified both in the config file and on the command line (e.g. `--batch_size=7`), and are resolved as follows (any source here overrides all the preceding ones):

* `argparse` defaults — these are specified in the code directly;
* those saved in a loaded checkpoint (if starting from a checkpoint);
* your `--config` file;
* command line.

The command is

```bash
python3 train.py --config=config_name [any extra arguments ...]
```

Or, with multiple GPUs,

```bash
python3 -um torch.distributed.launch --nproc_per_node=<number of GPUs> train.py --config=config_name [any extra arguments ...]
```

## Reference

Consider citing us if you use the code:

```bibtex
@InProceedings{Burkov_2020_CVPR,
author = {Burkov, Egor and Pasechnik, Igor and Grigorev, Artur and Lempitsky, Victor},
title = {Neural Head Reenactment with Latent Pose Descriptors},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

