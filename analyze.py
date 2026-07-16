"""Post-session swing analysis of a CSV produced by main.py --csv.

Finds strokes by wrist-speed peaks, splits each into backswing / contact /
follow-through, prints joint angles at contact, and optionally scores each
stroke against a reference stroke with DTW.

Pure stdlib, runs anywhere (no Jetson needed):

  python3 analyze.py session.csv
  python3 analyze.py session.csv --reference good_forehand.csv
"""
import argparse
import csv
import math

ANGLE_KEYS = ("elbow", "shoulder", "knee", "hip")
# fraction of the peak wrist speed where a stroke starts/ends
STROKE_EDGE = 0.15


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "t": float(r["t"]),
                "speed": float(r["wrist_speed"]) if r.get("wrist_speed") else None,
                "angles": {k: float(r[k]) if r.get(k) else None for k in ANGLE_KEYS},
            })
    return rows


def moving_average(vals, window=5):
    out = []
    for i in range(len(vals)):
        chunk = [v for v in vals[max(0, i - window // 2):i + window // 2 + 1]
                 if v is not None]
        out.append(sum(chunk) / len(chunk) if chunk else 0.0)
    return out


def find_strokes(rows, min_gap_s=1.0):
    """Returns [(start_idx, peak_idx, end_idx)] for each detected stroke."""
    speed = moving_average([r["speed"] for r in rows])
    if not speed:
        return []
    global_max = max(speed)
    if global_max <= 0:
        return []
    threshold = 0.4 * global_max

    peaks = []
    for i in range(1, len(speed) - 1):
        if speed[i] >= threshold and speed[i] >= speed[i - 1] and speed[i] >= speed[i + 1]:
            if peaks and rows[i]["t"] - rows[peaks[-1]]["t"] < min_gap_s:
                if speed[i] > speed[peaks[-1]]:
                    peaks[-1] = i  # keep the stronger of two close peaks
            else:
                peaks.append(i)

    strokes = []
    for p in peaks:
        edge = STROKE_EDGE * speed[p]
        start = p
        while start > 0 and speed[start] > edge:
            start -= 1
        end = p
        while end < len(speed) - 1 and speed[end] > edge:
            end += 1
        strokes.append((start, p, end))
    return strokes


def angle_series(rows, start, end, keys=("elbow", "knee")):
    """Concatenated angle trajectories over a stroke, gaps forward-filled,
    normalized to [0..1] so joints weigh equally in DTW."""
    series = []
    for k in keys:
        prev = None
        for r in rows[start:end + 1]:
            v = r["angles"][k]
            if v is None:
                v = prev if prev is not None else 90.0
            prev = v
            series.append(v / 180.0)
    return series


def dtw_distance(a, b):
    """Plain O(n*m) DTW, normalized by path length."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    inf = float("inf")
    prev = [inf] * (m + 1)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = [inf] * (m + 1)
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[m] / (n + m)


def similarity_score(dist):
    """Map normalized DTW distance to a 0-100 score. The scale is empirical:
    ~0.005 distance = same stroke shape (~90%), ~0.07 = clearly different (~25%)."""
    return 100.0 * math.exp(-dist / 0.05)


def fmt_angle(v):
    return f"{v:5.1f}" if v is not None else "   --"


def main():
    ap = argparse.ArgumentParser(description="Swing analysis of a main.py CSV log")
    ap.add_argument("csv", help="session CSV from main.py --csv")
    ap.add_argument("--reference", default=None,
                    help="CSV with one good stroke to compare against (DTW)")
    args = ap.parse_args()

    rows = load(args.csv)
    strokes = find_strokes(rows)
    if not strokes:
        print("No strokes detected — is there a wrist_speed column with real motion?")
        return

    ref_series = None
    if args.reference:
        ref_rows = load(args.reference)
        ref_strokes = find_strokes(ref_rows)
        if not ref_strokes:
            print(f"Warning: no stroke found in reference {args.reference}, skipping DTW")
        else:
            # strongest stroke in the reference file is the etalon
            best = max(ref_strokes, key=lambda s: ref_rows[s[1]]["speed"] or 0)
            ref_series = angle_series(ref_rows, best[0], best[2])

    t0 = rows[0]["t"]
    header = f"{'#':>2}  {'time':>6}  {'backswing':>9}  {'follow':>6}  " \
             + "  ".join(f"{k:>8}" for k in ANGLE_KEYS)
    if ref_series:
        header += f"  {'match':>6}"
    print(f"{len(strokes)} strokes detected\n")
    print(header)
    print("(angles at contact, degrees; backswing/follow-through in seconds)")

    for n, (start, peak, end) in enumerate(strokes, 1):
        contact = rows[peak]
        line = (f"{n:>2}  {contact['t'] - t0:6.1f}  "
                f"{contact['t'] - rows[start]['t']:9.2f}  "
                f"{rows[end]['t'] - contact['t']:6.2f}  "
                + "  ".join(f"{fmt_angle(contact['angles'][k]):>8}" for k in ANGLE_KEYS))
        if ref_series:
            dist = dtw_distance(angle_series(rows, start, end), ref_series)
            line += f"  {similarity_score(dist):5.0f}%"
        print(line)


if __name__ == "__main__":
    main()
