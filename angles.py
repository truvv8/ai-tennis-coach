"""Joint angle math on top of trt_pose keypoints.

Keypoint indices follow human_pose.json from trt_pose:
0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle, 17 neck
"""
import math

NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


def angle_deg(a, b, c):
    """Angle ABC (at vertex b) in degrees, points are (x, y)."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def _tri(kps, i, j, k):
    if i in kps and j in kps and k in kps:
        return angle_deg(kps[i], kps[j], kps[k])
    return None


def compute_angles(kps, right_handed=True):
    """kps: {index: (x, y)} normalized coords. Returns dict of angles in degrees.

    For a right-handed player filmed from the side, the hitting arm is the right one.
    """
    if right_handed:
        sh, el, wr = R_SHOULDER, R_ELBOW, R_WRIST
        hip, knee, ankle = R_HIP, R_KNEE, R_ANKLE
    else:
        sh, el, wr = L_SHOULDER, L_ELBOW, L_WRIST
        hip, knee, ankle = L_HIP, L_KNEE, L_ANKLE

    angles = {
        # elbow bend of the hitting arm (180 = straight)
        "elbow": _tri(kps, sh, el, wr),
        # arm elevation relative to the torso
        "shoulder": _tri(kps, el, sh, hip),
        # front knee bend
        "knee": _tri(kps, hip, knee, ankle),
        "hip": _tri(kps, sh, hip, knee),
    }

    # shoulder-line tilt vs horizontal: proxy for shoulder turn on a side view
    if L_SHOULDER in kps and R_SHOULDER in kps:
        lx, ly = kps[L_SHOULDER]
        rx, ry = kps[R_SHOULDER]
        angles["shoulder_line"] = math.degrees(math.atan2(ry - ly, rx - lx))
    else:
        angles["shoulder_line"] = None

    return angles


def wrist_speed(prev_kps, kps, dt, right_handed=True):
    """Normalized wrist speed between two frames; used to find the swing peak."""
    wr = R_WRIST if right_handed else L_WRIST
    if prev_kps is None or wr not in prev_kps or wr not in kps or dt <= 0:
        return None
    (x0, y0), (x1, y1) = prev_kps[wr], kps[wr]
    return math.hypot(x1 - x0, y1 - y0) / dt
