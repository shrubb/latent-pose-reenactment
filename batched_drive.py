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
    # The directory where all fine-tuned checkpoints reside
    output_dir = Path(f"puppeteering/{MODEL_NAME}_{ITERATION}")
    assert output_dir.is_dir()

    # Identities to drive
    identities_to_drive = list(output_dir.iterdir()) # change this if you want, e.g.:
    identities_to_drive = [output_dir / string_to_valid_filename(x) for x in [
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
    ]]

    # Drivers
    DATASET_ROOT = Path("/Vol0/user/e.burkov/Shared/VoxCeleb2_30TestIdentities")
    # DATASET_ROOT = Path("/Vol0/user/e.burkov/Shared/Custom avatar drivers")
    # DATASET_ROOT = Path("/Vol0/user/e.burkov/Datasets/violet/VoxCeleb2_test_finetuning")

    IMAGES_DIR = DATASET_ROOT / ("images-cropped-ffhq" if MODEL_NAME == "Zakharov" else "images-cropped")
    DRIVERS = [
        "id00061/cAT9aR8oFx0/driver",
        "id00061/Df_m1slf_hY/driver",
        "id00812/XoAi2n4S2wo/driver",
        "id01106/B08yOvYMF7Y/driver",
        "id01228/7qHTvs0VO68/driver",
        "id01333/9kgJaduwKkY/driver",
        "id01437/4lFDvxXzYWY/driver",
        "id02057/s5VqJY7DDEE/driver",
        "id02548/x2LUQEUXdz4/driver",
        "id03127/uiRiyK8Qlic/driver",
        "id03178/cCoNRuzAL-A/driver",
        "id03178/fnARFfUwf2s/driver",
        "id03524/GkvScYvOJ7o/driver",
        "id03839/LhI_8AWX_Mg/driver",
        "id03839/PUwanP-C5qg/driver",
        "id03862/fsCqKQb9Rdg/driver",
        "id04094/JUYMzfVp8zI/driver",
        "id04950/PQEAck-3wcA/driver",
        "id05459/3TI6dVmEwzw/driver",
        "id05714/wFGNufaMbDY/driver",
        "id06104/7UnGAS5-jpU/driver",
        "id06811/KmvEwL3fP9Q/driver",
        "id07312/h1dszoDi1E8/driver",
        "id07663/54qlJ2HZ08s/driver",
        "id07802/BfQUBDw7TiM/driver",
        "id07868/JC0QT4oXh2Y/driver",
        "id07961/464OHFffwjI/driver",
        "id07961/hROZwL8pbGg/driver",
        "id08149/vxBFGKGXSFA/driver",
        "id08701/UeUyLqpLz70/driver",
    ]

    for identity_to_drive in identities_to_drive:
        # Get fine-tuned checkpoint
        checkpoint_path = identity_to_drive / "checkpoints"
        assert checkpoint_path.is_dir()
        all_checkpoints = sorted(checkpoint_path.iterdir())
        if len(all_checkpoints) > 1:
            print(
                f"WARNING: there are {len(all_checkpoints)} checkpoints in" \
                f"{checkpoint_path}, using the latest one ({all_checkpoints[-1]})")
        checkpoint_path = all_checkpoints[-1]

        command = [
            "python3",
            "drive.py",
            str(checkpoint_path),
            str(DATASET_ROOT),
            "--destination", str(identity_to_drive / "driving-results"),
            "--images_paths"] + DRIVERS

        if socket.gethostname() == 'airulsf01':
            # Submit to LSF
            job_name = f"driving_{identity_to_drive.name}_{MODEL_NAME}_{ITERATION}"

            if os.getenv('AIRUGPUB') is None or os.getenv('AIRUGPUA') is None:
                exec_hosts = ""
            else:
                exec_hosts = f"{os.getenv('AIRUGPUB')} {os.getenv('AIRUGPUA')}"
            # exec_hosts = "airugpub01 airugpub02 airugpub03 " + " ".join("airugpua%02d" % i for i in (1,3,4,5,6,7,8,9,10,11,12,13))
            command = [
                "bsub", "-J", str(job_name), "-gpu", "num=1:mode=exclusive_process",
                "-o", f"logs/{job_name}.txt", "-m", str(exec_hosts),
            ] + command

        subprocess.run(command)
