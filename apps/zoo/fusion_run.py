# apps/zoo/run_fusion.py
"""
Run a trained fusion model on a dict of head scores.
"""

import argparse
import json

from hoaxhertz.utils.fusion import FusionModel


def parse_kv(pairs):
    out = {}
    for p in pairs:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[k.strip()] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="artifacts/models/fusion/model.joblib")
    ap.add_argument("--json", type=str, default=None, help="Path to JSON with head scores.")
    ap.add_argument(
        "--kv",
        type=str,
        action="append",
        default=[],
        help="key=value pairs (e.g., ambient=0.7)",
    )
    args = ap.parse_args()

    fm = FusionModel.load(args.model)

    if args.json:
        with open(args.json, encoding="utf-8") as f:
            scores = json.load(f)
    else:
        scores = parse_kv(args.kv)

    prob = fm.predict_proba(scores)
    print(json.dumps({"scores": scores, "probability": prob}, indent=2))


if __name__ == "__main__":
    main()
