# Installing on a Jetson Nano

Target: Jetson Nano 4GB with JetPack 4.6.x (the last JetPack for the original Nano).

## 0. Flash (once, if the Nano is fresh)

Grab the JetPack 4.6.x SD card image from developer.nvidia.com and write it with balenaEtcher. First boot can be done headless: plug the microUSB into your laptop, `screen /dev/tty.usbmodem* 115200` (macOS) or `screen /dev/ttyACM0 115200` (Linux), create a user, join wifi/ethernet.

## 1. SSH in

```bash
ssh <user>@<jetson-ip>
head -n1 /etc/nv_tegra_release   # should say R32.x
sudo nvpmodel -m 0 && sudo jetson_clocks   # max performance
```

If you don't know the IP: on your laptop, ping-sweep your subnet and look for a host with port 22 open, or check the router's client list.

## 2. Run the setup script

```bash
git clone https://github.com/truvv8/ai-tennis-coach
cd ai-tennis-coach
bash setup.sh
```

It sets up swap, installs the NVIDIA PyTorch wheel, builds torchvision from source (this is the slow part, expect 30–60 min), installs torch2trt and trt_pose, downloads the pose model weights and converts them to a TensorRT engine. It's idempotent — if it dies halfway (usually the torchvision build running out of memory — that's what the swap is for), just run it again and it skips what's done.

If the weights download from Google Drive fails, grab `resnet18_baseline_att_224x224_A` manually from the link in the [trt_pose README](https://github.com/NVIDIA-AI-IOT/trt_pose), drop the .pth here and re-run.

## 3. Run

```bash
python3 main.py --source 0 --stream 8080     # usb camera
python3 main.py --source csi --stream 8080   # csi (raspberry pi) camera
```

Open `http://<jetson-ip>:8080/` in a browser on your laptop. Run it inside `tmux` so it survives a dropped ssh connection.

## Manual install

If you'd rather do what setup.sh does by hand, the steps are:

1. 4GB swapfile (`fallocate`, `mkswap`, `swapon`, fstab entry)
2. `apt install python3-pip libopenblas-base libjpeg-dev zlib1g-dev python3-matplotlib git`
3. PyTorch wheel for JetPack 4.6.1 from NVIDIA:
   `https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.11.0a0+17540c5+nv22.01-cp36-cp36m-linux_aarch64.whl`
4. torchvision v0.12.0 built from source (`git clone --branch v0.12.0 https://github.com/pytorch/vision`, `python3 setup.py install --user`)
5. [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt): `sudo python3 setup.py install --plugins`
6. [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose): `sudo python3 setup.py install`, copy `tasks/human_pose/human_pose.json` here, download the resnet18 weights
7. `python3 build_engine.py`
