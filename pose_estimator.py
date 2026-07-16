"""trt_pose inference wrapper for Jetson.

Expects a TensorRT-optimized model produced by build_engine.py (model_trt.pth)
and human_pose.json from the trt_pose repo next to this file.
"""
import json
import cv2
import torch

import trt_pose.coco
from trt_pose.parse_objects import ParseObjects
from torch2trt import TRTModule

INPUT_SIZE = 224  # resnet18_baseline_att_224x224_A


class PoseEstimator:
    def __init__(self, model_path="model_trt.pth", topology_path="human_pose.json"):
        with open(topology_path) as f:
            self.human_pose = json.load(f)
        self.topology = trt_pose.coco.coco_category_to_topology(self.human_pose)
        self.num_keypoints = len(self.human_pose["keypoints"])

        self.model = TRTModule()
        self.model.load_state_dict(torch.load(model_path))
        self.parse_objects = ParseObjects(self.topology)

        self.device = torch.device("cuda")
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)[:, None, None]
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device)[:, None, None]

    def _preprocess(self, frame_bgr):
        img = cv2.resize(frame_bgr, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).to(self.device).permute(2, 0, 1).float().div_(255.0)
        tensor.sub_(self.mean).div_(self.std)
        return tensor[None, ...]

    @torch.no_grad()
    def infer(self, frame_bgr):
        """Returns list of persons; each person is {kp_index: (x, y)} in
        normalized [0..1] coordinates of the original frame."""
        data = self._preprocess(frame_bgr)
        cmap, paf = self.model(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf)

        persons = []
        for i in range(int(counts[0])):
            obj = objects[0][i]
            kps = {}
            for j in range(self.num_keypoints):
                k = int(obj[j])
                if k >= 0:
                    peak = peaks[0][j][k]
                    # peaks are (y, x) normalized
                    kps[j] = (float(peak[1]), float(peak[0]))
            persons.append(kps)
        return persons

    def pick_main_person(self, persons):
        """The player = the person with the most detected keypoints
        (ties broken by bounding-box area)."""
        def score(kps):
            if not kps:
                return (0, 0.0)
            xs = [p[0] for p in kps.values()]
            ys = [p[1] for p in kps.values()]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            return (len(kps), area)

        return max(persons, key=score) if persons else None

    def draw(self, frame, kps, color=(0, 255, 0)):
        h, w = frame.shape[:2]
        pts = {j: (int(x * w), int(y * h)) for j, (x, y) in kps.items()}
        for j, p in pts.items():
            cv2.circle(frame, p, 4, color, -1)
        for k in range(self.topology.shape[0]):
            a = int(self.topology[k][2])
            b = int(self.topology[k][3])
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], color, 2)
        return frame
