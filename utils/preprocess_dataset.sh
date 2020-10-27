# Preprocess head images for training our head reenactment system.
#
# Usage:
#     1. Choose if you want to preprocess videos OR images. Preprocessing both at once isn't supported!
#     2. Lay out your media as e.g.
#
#        `$DATASET_ROOT`/
#            images/
#                monalisa/
#                    painting.jpg
#                id00017/
#                    5MkXgwdrmJw/
#                        001.jpg
#                        002.jpg
#                        ...
#
#        Or, if you want to preprocess videos,
#
#        `$DATASET_ROOT`/
#            videos/
#                monalisa/
#                    X.avi
#                id00017/
#                    5MkXgwdrmJw/
#                        Y.avi
#
#     3. Edit this file: set `DATASET_ROOT` and `IDENTITIES` (read below how).
#     4. Also, set `DO_...` variables to `true` or `false`.
#        Usually, you'll set `DO_CROP=true` and `DO_COMPUTE_SEGMENTATION=true`,
#        and will set `DO_DECODE_VIDEOS` depending on whether you have video data.
#     5. Run `$ cd utils; bash preprocess_dataset.sh [FIRST_IDX [LAST_IDX]]` (see explanation below).

set -e

# Please use an ABSOLUTE path here!
DATASET_ROOT="/Vol1/dbstore/datasets/violet/VoxCeleb2_test_finetuning"

# echo "Unnamed: 0,path" > $dataset_dir/split.csv

# Initialize `IDENTITIES` -- the list of folders (paths relative to $DATASET_ROOT/images
# or $DATASET_ROOT/videos), each containing raw images or one video of some person.
cd "$DATASET_ROOT/images" # or e.g. `"$DATASET_ROOT/videos"`
IDENTITIES=(*) # or `(*/*)`, or whatever else
cd -

# Alternatively, you can specify them manually, e.g.:
# IDENTITIES=(
#     "monalisa"
#     "id00017/5MkXgwdrmJw"
# )

# Specify the range (segment) of identities to process. Useful for parallelizing.
FIRST_IDX=${1:-0}
LAST_IDX=${2:-999999999}

echo "Got ${#IDENTITIES[@]} folders, will process from ${FIRST_IDX}-th to ${LAST_IDX}-th"

# Switch off (set to `false` or comment out) unnecessary operations
DO_DECODE_VIDEOS=\
false

DO_CROP=\
true
DO_COMPUTE_SEGMENTATION=\
true
DO_COMPUTE_LANDMARKS=\
false
DO_COMPUTE_POSE_3DMM=\
false

DO_CROP_FFHQ=\
false
DO_COMPUTE_SEGMENTATION_FFHQ=\
false

################################# Extract frames from encoded videos ################################################

if [ "$DO_DECODE_VIDEOS" = true ]; then
    i=0
    for IDENTITY in "${IDENTITIES[@]}"; do
        if (($i >= $FIRST_IDX && $i < $LAST_IDX)); then
            IMAGES_OUTPUT="$DATASET_ROOT/images/$IDENTITY"
            mkdir -p "$IMAGES_OUTPUT"
            ffmpeg -hide_banner -i "$DATASET_ROOT/videos/${IDENTITY}"* -q:v 2 "$IMAGES_OUTPUT/%05d.jpg"
        fi
        let "i += 1"
    done
fi

################################# Crop using only face detector ("latent pose style" crop) ############################

IMAGES_CROPPED_DIR_NAME="images-cropped"
KEYPOINTS_CROPPED_DIR_NAME="keypoints-cropped"
SEGMENTATION_DIR_NAME="segmentation-cropped"
POSE_3DMM_DIR_NAME="3dmm-descriptors"

if [ "$DO_CROP" = true ]; then
    i=0
    for IDENTITY in "${IDENTITIES[@]}"; do #$DATASET_ROOT/images/*; do
        if (($i >= $FIRST_IDX && $i < $LAST_IDX)); then
            # vid=$IDENTITY/$(ls -1 $IDENTITY | shuf -n1)
            vid="$DATASET_ROOT/images/$IDENTITY"
            filename="$(basename -- "$vid")"
            filename="${filename%.*}"
            dir="$(dirname "$vid")"
            dir="$(basename "$dir")"

            echo $i $dir $vid $filename
            # echo "$i,SAIC-selfie-videos/$dir/$filename" >> $DATASET_ROOT/split.csv

            mkdir -p "$DATASET_ROOT/$IMAGES_CROPPED_DIR_NAME/$IDENTITY/"

            if [ "$DO_COMPUTE_LANDMARKS" = true ]; then
                python3 crop_as_in_dataset.py --crop-style=latentpose --save-extra-data "$DATASET_ROOT/images/$IDENTITY/" "$DATASET_ROOT/$IMAGES_CROPPED_DIR_NAME/$IDENTITY/"
                mkdir -p "$DATASET_ROOT/$KEYPOINTS_CROPPED_DIR_NAME/$IDENTITY/"

                # Move '.npy' files to a separate folder
                find "$DATASET_ROOT/$IMAGES_CROPPED_DIR_NAME/$IDENTITY/" -type f -name "*.npy" -exec mv {} "$DATASET_ROOT/$KEYPOINTS_CROPPED_DIR_NAME/$IDENTITY/" \;
            else
                python3 crop_as_in_dataset.py --crop-style=latentpose "$DATASET_ROOT/images/$IDENTITY/" "$DATASET_ROOT/$IMAGES_CROPPED_DIR_NAME/$IDENTITY/"
            fi
        fi
        let "i += 1"
    done;
fi

# Compute segmentation
if [ "$DO_COMPUTE_SEGMENTATION" = true ]; then
    TMPFILE=$(mktemp)
    i=0
    for IDENTITY in "${IDENTITIES[@]}"; do
        if (($i >= $FIRST_IDX && $i < $LAST_IDX)); then
            for FILE in $(ls -1 "$DATASET_ROOT/$IMAGES_CROPPED_DIR_NAME/$IDENTITY/"); do
                echo "$DATASET_ROOT/$IMAGES_CROPPED_DIR_NAME/$IDENTITY/$FILE" >> "$TMPFILE"
            done
        fi
        let "i += 1"
    done

    SEGMENTATION_OUTPUT="$DATASET_ROOT/$SEGMENTATION_DIR_NAME"

    cd Graphonomy
    python3 exp/inference/inference_folder.py --images_path "$TMPFILE" --output_dir "$SEGMENTATION_OUTPUT" --common_prefix "$DATASET_ROOT/$IMAGES_CROPPED_DIR_NAME" --model_path data/model/universal_trained.pth --tta 0.75,1.0,1.5,2.0
    cd -
fi

# Compute 3DMM pose+expression vectors
if [ "$DO_COMPUTE_POSE_3DMM" = true ]; then
    TMPFILE=$(mktemp)
    i=0
    for IDENTITY in "${IDENTITIES[@]}"; do
        if (($i >= $FIRST_IDX && $i < $LAST_IDX)); then
            for FILE in $(ls -1 "$DATASET_ROOT/$IMAGES_CROPPED_DIR_NAME/$IDENTITY/"); do
                echo "$DATASET_ROOT/$IMAGES_CROPPED_DIR_NAME/$IDENTITY/$FILE" >> "$TMPFILE"
            done
        fi
        let "i += 1"
    done

    POSE_3DMM_OUTPUT="$DATASET_ROOT/$POSE_3DMM_DIR_NAME"

    cd /Vol0/user/e.burkov/Projects/Expression-Net
    python2 compute_3DMM_coefficients_noBboxes.py --images_path "$TMPFILE" --output_dir "$POSE_3DMM_OUTPUT" --common_prefix "$DATASET_ROOT/$IMAGES_CROPPED_DIR_NAME"
    cd -
fi

###################################### Crop using landmarks ("FFHQ style" crop) #######################################

IMAGES_CROPPED_DIR_NAME="images-cropped-ffhq"
KEYPOINTS_CROPPED_DIR_NAME="keypoints-cropped-ffhq"
SEGMENTATION_DIR_NAME="segmentation-cropped-ffhq"

if [ "$DO_CROP_FFHQ" = true ]; then
    i=0
    for IDENTITY in "${IDENTITIES[@]}"; do #$DATASET_ROOT/images/*; do
        if (($i >= $FIRST_IDX && $i < $LAST_IDX)); then
            # vid=$IDENTITY/$(ls -1 $IDENTITY | shuf -n1)
            vid="$DATASET_ROOT/images/$IDENTITY"
            filename="$(basename -- "$vid")"
            filename="${filename%.*}"
            dir="$(dirname "$vid")"
            dir="$(basename "$dir")"

            echo $i $dir $vid $filename
            # echo "$i,SAIC-selfie-videos/$dir/$filename" >> $DATASET_ROOT/split.csv

            mkdir -p "$DATASET_ROOT/$IMAGES_CROPPED_DIR_NAME/$IDENTITY/"
            python3 crop_as_in_dataset.py --crop-style=ffhq --save-extra-data "$DATASET_ROOT/images/$IDENTITY/" "$DATASET_ROOT/$IMAGES_CROPPED_DIR_NAME/$IDENTITY/"
            mkdir -p "$DATASET_ROOT/$KEYPOINTS_CROPPED_DIR_NAME/$IDENTITY/"

            # Move '.npy' files to a separate folder
            find "$DATASET_ROOT/$IMAGES_CROPPED_DIR_NAME/$IDENTITY/" -type f -name "*.npy" -exec mv {} "$DATASET_ROOT/$KEYPOINTS_CROPPED_DIR_NAME/$IDENTITY/" \;
        fi
        let "i += 1"
    done;
fi

# Compute segmentation
if [ "$DO_COMPUTE_SEGMENTATION_FFHQ" = true ]; then
    TMPFILE=$(mktemp)
    i=0
    for IDENTITY in "${IDENTITIES[@]}"; do
        if (($i >= $FIRST_IDX && $i < $LAST_IDX)); then
            for FILE in $(ls -1 "$DATASET_ROOT/$IMAGES_CROPPED_DIR_NAME/$IDENTITY/"); do
                echo "$DATASET_ROOT/$IMAGES_CROPPED_DIR_NAME/$IDENTITY/$FILE" >> "$TMPFILE"
            done
        fi
        let "i += 1"
    done

    SEGMENTATION_OUTPUT="$DATASET_ROOT/$SEGMENTATION_DIR_NAME"

    cd Graphonomy
    python3 exp/inference/inference_folder.py --images_path "$TMPFILE" --output_dir "$SEGMENTATION_OUTPUT" --common_prefix "$DATASET_ROOT/$IMAGES_CROPPED_DIR_NAME" --model_path data/model/universal_trained.pth --tta 0.75,1.0,1.5,2.0
    cd -
fi
