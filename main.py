"""AI Tennis Coach — starter pipeline for Jetson Nano.

Camera -> trt_pose skeleton -> joint angles -> on-frame feedback,
viewable in a browser via MJPEG (works headless over SSH).

Examples:
  python3 main.py --source 0 --stream 8080          # USB camera
  python3 main.py --source csi --stream 8080        # CSI (Raspberry Pi) camera
  python3 main.py --source clip.mp4 --csv out.csv   # analyze a recorded clip
"""
import argparse
import csv
import time

import cv2

from angles import compute_angles, wrist_speed
from pose_estimator import PoseEstimator
from streamer import MJPEGStreamer

# Rough form checks for a side-view forehand; tune against your own reference clips.
FEEDBACK_RULES = [
    ("elbow", lambda a: a is not None and a < 90,
     "Elbow too bent — extend through contact"),
    ("knee", lambda a: a is not None and a > 165,
     "Legs too straight — bend the front knee"),
]


def gst_csi_pipeline(width=1280, height=720, fps=30):
    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1, format=NV12 ! "
        "nvvidconv flip-method=0 ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )


def open_source(source):
    if source == "csi":
        return cv2.VideoCapture(gst_csi_pipeline(), cv2.CAP_GSTREAMER)
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap
    return cv2.VideoCapture(source)  # video file


def put_lines(frame, lines, org=(10, 30), color=(255, 255, 255)):
    x, y = org
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 28
    return y


def main():
    ap = argparse.ArgumentParser(description="AI Tennis Coach (Jetson Nano)")
    ap.add_argument("--source", default="0", help="'csi', camera index, or video file")
    ap.add_argument("--stream", type=int, default=8080, help="MJPEG port (0 = off)")
    ap.add_argument("--csv", default=None, help="log angles per frame to CSV")
    ap.add_argument("--left-handed", action="store_true")
    ap.add_argument("--model", default="model_trt.pth")
    args = ap.parse_args()

    pose = PoseEstimator(model_path=args.model)
    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    streamer = None
    if args.stream:
        streamer = MJPEGStreamer(args.stream)
        streamer.start()

    csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["t", "elbow", "shoulder", "knee", "hip",
                             "shoulder_line", "wrist_speed"])

    right_handed = not args.left_handed
    prev_kps, prev_t = None, None
    fps, fps_t, fps_n = 0.0, time.time(), 0

    print("Running. Ctrl+C to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("End of stream.")
                break
            now = time.time()

            persons = pose.infer(frame)
            player = pose.pick_main_person(persons)

            lines = [f"FPS: {fps:.1f}  persons: {len(persons)}"]
            if player:
                pose.draw(frame, player)
                ang = compute_angles(player, right_handed=right_handed)
                speed = wrist_speed(prev_kps, player, now - (prev_t or now),
                                    right_handed=right_handed)
                for name in ("elbow", "shoulder", "knee", "hip"):
                    v = ang[name]
                    lines.append(f"{name}: {v:5.1f}" if v is not None else f"{name}: --")

                y = put_lines(frame, lines)
                for key, is_bad, msg in FEEDBACK_RULES:
                    if is_bad(ang[key]):
                        y = put_lines(frame, [msg], org=(10, y), color=(0, 0, 255))

                if csv_writer:
                    csv_writer.writerow([
                        f"{now:.3f}",
                        *(f"{ang[k]:.1f}" if ang[k] is not None else "" for k in
                          ("elbow", "shoulder", "knee", "hip", "shoulder_line")),
                        f"{speed:.3f}" if speed is not None else "",
                    ])
                prev_kps, prev_t = player, now
            else:
                put_lines(frame, lines + ["no player detected"])

            if streamer:
                streamer.update(frame)

            fps_n += 1
            if now - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        if csv_writer:
            csv_file.close()


if __name__ == "__main__":
    main()
