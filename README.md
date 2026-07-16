# 🎾 AI Tennis Coach

**Real-time tennis technique analysis on a $99 Jetson Nano.** Point a camera at the court, get instant feedback on your form — no cloud, no subscription, all inference runs on-device.

[Русская версия / Russian version](README.ru.md)

```
camera ──▶ trt_pose (TensorRT, ~20 FPS) ──▶ joint angles ──▶ live feedback
                                                │
                                                └──▶ browser stream + CSV log
```

## What it does

- **Skeleton tracking** of the player using [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose), NVIDIA's pose model optimized for Jetson (ResNet18 + TensorRT FP16)
- **Joint angles every frame**: hitting-arm elbow and shoulder, front knee, hip, shoulder-line tilt
- **Rule-based form feedback** rendered on the video ("Elbow too bent — extend through contact")
- **Headless-friendly**: built-in zero-dependency MJPEG server — watch the annotated stream in any browser while the Jetson runs over SSH
- **Swing analysis offline**: run on a recorded clip, log angles + wrist speed to CSV, find the contact point by the wrist-speed peak

## Hardware

| Part | Notes |
|---|---|
| Jetson Nano 4GB | JetPack 4.6.x |
| Camera | USB webcam or CSI (Raspberry Pi cam) |
| Placement | Side view, perpendicular to the stroke, waist height, full body in frame |

## Quick start

On the Jetson (PyTorch for Jetson, torch2trt and trt_pose installed — see [setup guide](README.ru.md)):

```bash
git clone https://github.com/truvv8/ai-tennis-coach
cd ai-tennis-coach

# get the model
cp ~/trt_pose/tasks/human_pose/human_pose.json .
# download resnet18_baseline_att_224x224_A_epoch_249.pth (link in the trt_pose README)

python3 build_engine.py        # one-time TensorRT conversion, ~5-10 min

python3 main.py --source 0 --stream 8080     # USB camera
python3 main.py --source csi --stream 8080   # CSI camera
```

Open `http://<jetson-ip>:8080/` in a browser — live video with skeleton, angles and coaching cues.

Analyze a recorded stroke:

```bash
python3 main.py --source forehand.mp4 --csv forehand.csv --stream 0
```

## Project layout

| File | Purpose |
|---|---|
| `main.py` | capture → pose → angles → feedback → stream loop |
| `pose_estimator.py` | trt_pose wrapper: inference, player selection, skeleton drawing |
| `angles.py` | joint-angle math, wrist speed |
| `build_engine.py` | one-time PyTorch → TensorRT conversion |
| `streamer.py` | dependency-free MJPEG server for headless viewing |

## Tuning the feedback

The thresholds in `FEEDBACK_RULES` (main.py) are starting points for a side-view forehand. Record a few reference strokes with `--csv`, look at the real angle values at contact, and adjust. Left-handed players: pass `--left-handed`.

## Limitations

A single 2D camera can't measure depth — racket-face angle and wrist pronation are out of reach; this catches the big form errors (collapsed elbow, no knee bend, late shoulder turn). Stereo or an IMU on the racket would be the next step.

## Roadmap

- [ ] Stroke phase detection (backswing → contact → follow-through) from the wrist-speed profile
- [ ] DTW comparison against a reference stroke
- [ ] Serve and backhand rule sets
- [ ] Audio cues between points

## License

MIT
