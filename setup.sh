#!/usr/bin/env bash
# One-shot setup for ai-tennis-coach on a Jetson Nano, JetPack 4.6.x.
# Run on the Jetson itself: bash setup.sh
# Takes a while (torchvision builds from source, ~30-60 min on a Nano).
set -euo pipefail

TORCH_WHL="torch-1.11.0a0+17540c5+nv22.01-cp36-cp36m-linux_aarch64.whl"
TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/${TORCH_WHL}"
TORCHVISION_VERSION="v0.12.0"   # the pairing for torch 1.11 per NVIDIA's PyTorch-for-Jetson thread
WEIGHTS="resnet18_baseline_att_224x224_A_epoch_249.pth"
WEIGHTS_GDRIVE_ID="1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd"   # from the trt_pose README

step() { echo; echo "==== $* ===="; }

if [ ! -f /etc/nv_tegra_release ]; then
    echo "No /etc/nv_tegra_release found — run this on the Jetson, not your laptop."
    exit 1
fi
step "Jetson detected: $(head -c60 /etc/nv_tegra_release)"

step "Swap (needed for the torchvision build)"
if ! swapon --show | grep -q .; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
else
    echo "swap already active, skipping"
fi

step "APT packages"
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-base libjpeg-dev zlib1g-dev \
    python3-matplotlib git

step "PyTorch (NVIDIA wheel for JetPack 4.6)"
if python3 -c "import torch" 2>/dev/null; then
    echo "torch already installed: $(python3 -c 'import torch; print(torch.__version__)')"
else
    wget -nc "$TORCH_URL"
    pip3 install numpy "$TORCH_WHL"
fi
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'"
echo "torch OK, CUDA available"

step "torchvision ${TORCHVISION_VERSION} (source build, the long part)"
if python3 -c "import torchvision" 2>/dev/null; then
    echo "torchvision already installed, skipping"
else
    [ -d ~/torchvision-src ] || git clone --branch "$TORCHVISION_VERSION" --depth 1 \
        https://github.com/pytorch/vision ~/torchvision-src
    (cd ~/torchvision-src && export BUILD_VERSION="${TORCHVISION_VERSION#v}" \
        && python3 setup.py install --user)
fi

step "torch2trt"
if python3 -c "import torch2trt" 2>/dev/null; then
    echo "already installed, skipping"
else
    [ -d ~/torch2trt ] || git clone https://github.com/NVIDIA-AI-IOT/torch2trt ~/torch2trt
    (cd ~/torch2trt && sudo python3 setup.py install --plugins)
fi

step "trt_pose"
if python3 -c "import trt_pose" 2>/dev/null; then
    echo "already installed, skipping"
else
    sudo pip3 install tqdm cython pycocotools
    [ -d ~/trt_pose ] || git clone https://github.com/NVIDIA-AI-IOT/trt_pose ~/trt_pose
    (cd ~/trt_pose && sudo python3 setup.py install)
fi
cp -n ~/trt_pose/tasks/human_pose/human_pose.json . 2>/dev/null || true

step "Model weights"
if [ -f "$WEIGHTS" ]; then
    echo "already downloaded, skipping"
else
    pip3 install gdown || true
    if ! python3 -m gdown "https://drive.google.com/uc?id=${WEIGHTS_GDRIVE_ID}" -O "$WEIGHTS"; then
        echo "gdown failed — download manually from the link in the trt_pose README"
        echo "(resnet18_baseline_att_224x224_A) and put ${WEIGHTS} in this directory,"
        echo "then re-run this script."
        exit 1
    fi
fi

step "TensorRT engine"
if [ -f model_trt.pth ]; then
    echo "model_trt.pth already built, skipping"
else
    python3 build_engine.py
fi

step "Done"
echo "Try it:  python3 main.py --source 0 --stream 8080"
echo "Then open http://<jetson-ip>:8080/ in a browser."
