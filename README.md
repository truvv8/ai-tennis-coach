# ai-tennis-coach

Tennis stroke analysis on a Jetson Nano. You put a camera on the side of the court, it tracks your skeleton with [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose), computes joint angles every frame and overlays feedback on the video when your form is off (elbow collapsed at contact, legs too straight, that kind of thing). Everything runs on the Nano itself, nothing goes to the cloud.

There's a small MJPEG server built in, so you can run the whole thing headless over ssh and just open `http://<jetson-ip>:8080` in a browser to watch the annotated stream.

[Гайд по установке на русском](README.ru.md)

## Running it

You need PyTorch for Jetson (the NVIDIA wheel, not pip), torch2trt and trt_pose installed first — the [russian readme](README.ru.md) has the full walkthrough with commands.

```bash
git clone https://github.com/truvv8/ai-tennis-coach
cd ai-tennis-coach

cp ~/trt_pose/tasks/human_pose/human_pose.json .
# also download resnet18_baseline_att_224x224_A_epoch_249.pth,
# the link is in the trt_pose readme

python3 build_engine.py    # converts the model to TensorRT, one time, ~5-10 min

python3 main.py --source 0 --stream 8080     # usb camera
python3 main.py --source csi --stream 8080   # csi camera
```

To analyze a recorded clip instead and dump the angles to csv:

```bash
python3 main.py --source forehand.mp4 --csv forehand.csv --stream 0
```

The wrist speed column is useful for finding the contact point — it peaks right around contact.

## Files

- `main.py` — the main loop: capture, pose, angles, feedback, stream
- `pose_estimator.py` — trt_pose wrapper, picks the player out of detected people, draws the skeleton
- `angles.py` — joint angle math
- `build_engine.py` — one-time pytorch → tensorrt conversion
- `streamer.py` — the mjpeg server, no dependencies

## Camera placement

Side view, roughly perpendicular to the stroke, waist height, whole body in frame at contact. Side view is what makes elbow/knee/shoulder-turn readable — top view is better for ball tracking but useless for form.

## Tuning

The thresholds in `FEEDBACK_RULES` in main.py are rough starting points for a side-view forehand. Record a few of your own good strokes with `--csv`, look at what the angles actually are at contact, and adjust. If you're left-handed, pass `--left-handed`.

## Limitations

It's one 2D camera, so no depth — racket face angle and wrist pronation are not measurable. This catches the big form errors, not the subtle ones. Stereo or an IMU on the racket would fix that but that's a different project.

Things I want to add: stroke phase detection from the wrist speed profile, DTW comparison against a reference stroke, rule sets for serve and backhand.

MIT license.
