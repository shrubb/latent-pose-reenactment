# Feel totally free to modify any variables with capital names
import subprocess
from pathlib import Path
import os
import socket

def string_to_valid_filename(x):
    return x.replace('/', '_')

MODELS = [
    ("X2Face_vanilla", "00000009"),
    ("X2FacePretrained_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBug_Graphonomy", "01497847"),
    ("ExpressionNet_ResNeXt_3xVGGLossWeight_256_bboxes_noBottleneck_Graphonomy", "01080152"),
    ("FAbNetPretrained_ResNeXt_3xVGGLossWeight_bboxes_augScaleXNoShift_Graphonomy_smallerCrop", "01327623"),
    ("Zakharov", "01529383"),
    ("Zakharov_bboxes_vectorPose_noLandmarks_FineTune7xWeightNewMLP", "01464169"),

    # ("Zakharov_bboxes_vectorPose_noLandmarks", "01363326"),
    # ("ResNeXt_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBottleneck", "02275845"),
    # ("ResNeXt_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShiftReally_noBottleneck", "02023609"),
    # ("X2FacePretrained_ResNeXt_3xVGGLossWeight_256_bboxes_noBug_Graphonomy", "01337493"),
    # ("FAbNetPretrained_ResNeXt_3xVGGLossWeight_bboxes_Graphonomy_smallerCrop", "01222652"),

    # ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_aug_noBottleneck", "02492466"),
    # ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_aug_noBottleneck_cleanData", "01303150"),
    # ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBottleneck", "02227532"),
    # ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_augNoShift_noBottleneck", "02444553"),
    # ("ResNeXt_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBottleneck_noSegm", "01361933"),
    # ("ResNeXt_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBottleneck_noSegm", "01613709"),
    # ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_aug_noBottleneck", "02713859"),

    ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_aug_noBottleneck_FineTune7xWeight", "02714183"),
    ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBottleneck_noSegm", "02359800"),
    ("MobileNetV2_ResNeXt_3xVGGLossWeight_64_bboxes_aug_noBottleneck", "02191987"),
    ("MobileNetV2_ResNeXt_7xVGGLossWeight_256_bboxes_SAIC0.02_FromLearned", "02742693"),
    ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_aug_noBottleneck_cleanData", "01607204"),
    ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_augNoShift_noBottleneck", "02737273"),
    ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_augScaleXNoShift_noBottleneck", "02467652"),
    ("MobileNetV2_ResNeXt_3xVGGLossWeight_256_bboxes_augNoShift_noBottleneck_FineTune7xWeight", "02730111"),
]

for MODEL_NAME, ITERATION in MODELS:
    CHECKPOINT_PATH = f"experiments/{MODEL_NAME}/checkpoints/model_{ITERATION}.pth"
    assert Path(CHECKPOINT_PATH).is_file(), CHECKPOINT_PATH

    output_dir = Path(f"puppeteering/{MODEL_NAME}_{ITERATION}")

    DATASET_ROOT = Path("/Vol0/user/e.burkov/Shared/VoxCeleb2_30TestIdentities")
    # DATASET_ROOT = Path("/Vol0/user/e.burkov/Datasets/violet/VoxCeleb2_test_finetuning")
    # DATASET_ROOT = Path("/Vol0/user/e.burkov/Shared/Identity sources")

    IMAGES_DIR = DATASET_ROOT / "images-cropped"

    IDENTITIES = [
        "id00061/cAT9aR8oFx0/identity",
        "id00061/Df_m1slf_hY/identity",
        "id00812/XoAi2n4S2wo/identity",
        "id01106/B08yOvYMF7Y/identity",
        "id01228/7qHTvs0VO68/identity",
        "id01333/9kgJaduwKkY/identity",
        "id01437/4lFDvxXzYWY/identity",
        "id02057/s5VqJY7DDEE/identity",
        "id02548/x2LUQEUXdz4/identity",
        "id03127/uiRiyK8Qlic/identity",
        "id03178/cCoNRuzAL-A/identity",
        "id03178/fnARFfUwf2s/identity",
        "id03524/GkvScYvOJ7o/identity",
        "id03839/LhI_8AWX_Mg/identity",
        "id03839/PUwanP-C5qg/identity",
        "id03862/fsCqKQb9Rdg/identity",
        "id04094/JUYMzfVp8zI/identity",
        "id04950/PQEAck-3wcA/identity",
        "id05459/3TI6dVmEwzw/identity",
        "id05714/wFGNufaMbDY/identity",
        "id06104/7UnGAS5-jpU/identity",
        "id06811/KmvEwL3fP9Q/identity",
        "id07312/h1dszoDi1E8/identity",
        "id07663/54qlJ2HZ08s/identity",
        "id07802/BfQUBDw7TiM/identity",
        "id07868/JC0QT4oXh2Y/identity",
        "id07961/464OHFffwjI/identity",
        "id07961/hROZwL8pbGg/identity",
        "id08149/vxBFGKGXSFA/identity",
        "id08701/UeUyLqpLz70/identity",
    ]

    for identity in IDENTITIES:
        experiment_name = string_to_valid_filename(identity)
        checkpoint_output_dir = output_dir / experiment_name
        checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
        if (checkpoint_output_dir / 'checkpoints').is_dir() and len(list((checkpoint_output_dir / 'checkpoints').iterdir())) > 0:
            print(f"Skipping {checkpoint_output_dir}")
            continue

        num_images = sum(1 for _ in (IMAGES_DIR / identity).iterdir())
        MAX_BATCH_SIZE = 7 # 8 is memory limit for MobileNetV2+ResNeXt50 on P100
        batch_size = min(num_images, MAX_BATCH_SIZE)

        TARGET_NUM_ITERATIONS = 560
        iterations_in_epoch = num_images // batch_size
        num_epochs = (TARGET_NUM_ITERATIONS + iterations_in_epoch - 1) // iterations_in_epoch

        command = [
            "python3",
            "train.py",
            "--config", "finetuning-base",
            "--checkpoint_path", str(CHECKPOINT_PATH),
            "--data_root", str(DATASET_ROOT),
            "--train_split_path", str(identity),
            "--batch_size", str(batch_size),
            "--num_epochs", str(num_epochs),
            "--experiments_dir", str(output_dir),
            "--experiment_name", str(experiment_name),
            "--criterions", "adversarial, featmat, idt_embed, perceptual" + ", dice" * ('noSegm' not in MODEL_NAME and MODEL_NAME != "Zakharov"),
        ]

        if MODEL_NAME == "Zakharov":
            command += [
            "--img_dir", "images-cropped-ffhq",
            "--kp_dir", "keypoints-cropped-ffhq",
        ]

        if socket.gethostname() == 'airulsf01':
            # Submit to LSF
            job_name = f"{experiment_name}_{MODEL_NAME}_{ITERATION}"

            if os.getenv('AIRUGPUB') is None or os.getenv('AIRUGPUA') is None:
                exec_hosts = ""
            else:
                exec_hosts = f"{os.getenv('AIRUGPUB')} {os.getenv('AIRUGPUA')}"
            # exec_hosts = " ".join("airugpua%02d" % i for i in (1,3,4,5,6,7,8,9,10,11,12,13)) + " airugpub01 airugpub02 airugpub03"
            command = [
                "bsub", "-J", str(job_name), "-gpu", "num=1:mode=exclusive_process",
                "-o", f"logs/{job_name}.txt", "-m", str(exec_hosts),
            ] + command

        subprocess.run(command)
