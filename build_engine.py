"""One-time step on the Jetson: convert the PyTorch trt_pose checkpoint to a
TensorRT engine (model_trt.pth). Takes ~5-10 minutes on a Nano.

Prerequisites (see README): trt_pose + torch2trt installed,
resnet18_baseline_att_224x224_A_epoch_249.pth and human_pose.json downloaded.
"""
import json
import torch
import trt_pose.coco
import trt_pose.models
from torch2trt import torch2trt

WEIGHTS = "resnet18_baseline_att_224x224_A_epoch_249.pth"
OUTPUT = "model_trt.pth"
SIZE = 224

with open("human_pose.json") as f:
    human_pose = json.load(f)

num_parts = len(human_pose["keypoints"])
num_links = len(human_pose["skeleton"])

print("Loading PyTorch model...")
model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
model.load_state_dict(torch.load(WEIGHTS))

print("Converting to TensorRT (fp16), this takes a few minutes on a Nano...")
data = torch.zeros((1, 3, SIZE, SIZE)).cuda()
model_trt = torch2trt(model, [data], fp16_mode=True, max_workspace_size=1 << 25)

torch.save(model_trt.state_dict(), OUTPUT)
print(f"Saved {OUTPUT}")
