Setup
====

## Libraries

Guaranteed to work on Ubuntu 16.04 with Python 3.7+, CUDA 10.1 and also:

```bash
pip3 install \
    torch==1.5.1 \
    torchvision==0.6.1

pip3 install \
    tensorboard==2.2.2 \
    pandas==1.0.5 \
    torchsummary==1.5.1 \
    thop==0.0.31-2005241907 \
    tqdm==4.47.0 \
    scipy==1.5.1 \
    opencv-python==4.3.0.36 \
    imageio==2.9.0 \
    imageio-ffmpeg==0.4.2 \
    pymemcache==3.2.0 \
    python-memcached==1.59 \
    matplotlib==3.2.2 \
    seaborn==0.10.1 \
    huepy==1.2.1 \
    albumentations==0.4.6 \
    imgaug==0.4.0 \
    PyYAML==5.3.1 \
    munch==2.5.0 \
    pickle5==0.0.11 \
    jpeg4py==0.1.4 \
    face-alignment==1.1.1 \
    mxnet-cu101mkl==1.6.0.post0 \
    scikit-learn==0.23.2

pip3 install git+https://github.com/DmitryUlyanov/yamlenv.git
pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git
pip3 install git+https://github.com/dniku/insightface.git@763060b2
```

## Repository

1. `git submodule update --init`
1. [Download](https://drive.google.com/file/d/1sWJ54lCBFnzCNz5RTCGQmkVovkY9x8_D/view) [Graphonomy](https://github.com/Gaoyiminggithub/Graphonomy)'s `'universal_trained.pth'` and put it to `'utils/Graphonomy/data/model'`.
1. [Download](https://drive.google.com/drive/folders/1el_n1nSkxTpYO2yhAW4mseelFrlpn0Vh?usp=sharing) the two VGG weight files for perceptual losses and put them to `'criterions/common/'`.
