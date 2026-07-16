# ai-tennis-coach

[![lint](https://github.com/truvv8/ai-tennis-coach/actions/workflows/lint.yml/badge.svg)](https://github.com/truvv8/ai-tennis-coach/actions/workflows/lint.yml)

Tennis stroke analysis on a Jetson Nano. You put a camera on the side of the court, it tracks your skeleton with [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose), computes joint angles every frame and overlays feedback on the video when your form is off (elbow collapsed at contact, legs too straight, that kind of thing). Everything runs on the Nano itself, nothing goes to the cloud.

There's a small MJPEG server built in, so you can run the whole thing headless over ssh and just open `http://<jetson-ip>:8080` in a browser to watch the annotated stream.

[Гайд по установке на русском](README.ru.md)

## Running it

On the Jetson:

```bash
git clone https://github.com/truvv8/ai-tennis-coach
cd ai-tennis-coach
bash setup.sh    # swap, pytorch wheel, torch2trt, trt_pose, weights, tensorrt engine
                 # slow the first time (torchvision builds from source), idempotent

python3 main.py --source 0 --stream 8080     # usb camera
python3 main.py --source csi --stream 8080   # csi camera
```

Step-by-step version (flashing, ssh, manual install): [INSTALL.md](INSTALL.md).

Useful flags: `--record out.mp4` saves the annotated video, `--csv session.csv` logs angles and wrist speed per frame, `--left-handed` flips the hitting arm, `--no-smooth` disables the keypoint EMA filter.

## Analyzing a session

Log a session to csv, then:

```bash
python3 analyze.py session.csv --reference good_forehand.csv
```

It finds strokes by wrist-speed peaks, splits each into backswing / contact / follow-through and prints the joint angles at contact:

```
3 strokes detected

 #    time  backswing  follow     elbow  shoulder      knee       hip   match
 1     2.6       0.27    0.27     159.9     100.0     131.3     170.0     90%
 2     6.8       0.27    0.27      96.4      92.5     131.2     171.6     26%
 3    10.5       0.27    0.27     155.0      95.6     129.2     172.7     91%
```

The `match` column is a DTW comparison of the angle trajectories against a reference stroke — record one stroke you're happy with, save its csv, and every session gets scored against it. Stroke #2 above is a collapsed elbow: 96° at contact instead of ~157°, and the shape mismatch shows up in the score. `analyze.py` is pure stdlib and runs anywhere, not just the Jetson.

## Files

- `main.py` — the main loop: capture, pose, angles, feedback, stream
- `pose_estimator.py` — trt_pose wrapper, picks the player out of detected people, draws the skeleton
- `angles.py` — joint angle math
- `smoothing.py` — EMA filter for keypoints (raw pose output jitters ~±10° on angles)
- `analyze.py` — offline session analysis: stroke detection, phases, DTW scoring
- `build_engine.py` — one-time pytorch → tensorrt conversion
- `streamer.py` — the mjpeg server, no dependencies
- `setup.sh` — full Jetson setup in one script

## Camera placement

Side view, roughly perpendicular to the stroke, waist height, whole body in frame at contact. Side view is what makes elbow/knee/shoulder-turn readable — top view is better for ball tracking but useless for form.

## Tuning

The thresholds in `FEEDBACK_RULES` in main.py are rough starting points for a side-view forehand. Record a few of your own good strokes with `--csv`, look at what the angles actually are at contact, and adjust. If you're left-handed, pass `--left-handed`.

## Limitations

It's one 2D camera, so no depth — racket face angle and wrist pronation are not measurable. This catches the big form errors, not the subtle ones. Stereo or an IMU on the racket would fix that but that's a different project.

Things I want to add: rule sets for serve and backhand, a demo gif, audio cues between points.

MIT license.
