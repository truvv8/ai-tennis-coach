"""Build a reference stroke template from CSVs of good strokes.

Feed it sessions recorded with main.py --csv where the strokes are ones you're
happy with (your coach, your best day, a pro clip run through the pipeline).
Every stroke found in every file goes into the template: per-joint mean
trajectory plus tolerance band. The more strokes, the better.

  python3 train_reference.py good1.csv good2.csv -o forehand.json

Then score sessions against it:

  python3 analyze.py session.csv --reference forehand.json
"""
import argparse
import json
import math

from analyze import ANGLE_KEYS, N_SAMPLES, find_strokes, load, resample


def main():
    ap = argparse.ArgumentParser(description="Build a reference stroke template")
    ap.add_argument("csvs", nargs="+", help="CSV logs containing good strokes")
    ap.add_argument("-o", "--output", default="reference.json")
    args = ap.parse_args()

    trajectories = {k: [] for k in ANGLE_KEYS}
    total = 0
    for path in args.csvs:
        rows = load(path)
        strokes = find_strokes(rows)
        print(f"{path}: {len(strokes)} strokes")
        total += len(strokes)
        for start, _, end in strokes:
            for k in ANGLE_KEYS:
                trajectories[k].append(resample(rows, start, end, k))

    if total < 2:
        raise SystemExit("Need at least 2 strokes to estimate a tolerance band.")

    template = {"strokes": total, "keys": list(ANGLE_KEYS)}
    for k in ANGLE_KEYS:
        strokes_k = trajectories[k]
        mean, std = [], []
        for i in range(N_SAMPLES):
            col = [s[i] for s in strokes_k]
            m = sum(col) / len(col)
            mean.append(m)
            std.append(math.sqrt(sum((v - m) ** 2 for v in col) / (len(col) - 1)))
        template[k] = {"mean": mean, "std": std}

    with open(args.output, "w") as f:
        json.dump(template, f)
    print(f"Template from {total} strokes -> {args.output}")


if __name__ == "__main__":
    main()
