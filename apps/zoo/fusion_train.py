# apps/zoo/train_fusion.py
"""
Train an interpretable late-fusion model on head scores.
Input: CSV with columns [label, ambient?, enf?, mfcc?, locutor?, ...]
"""

import argparse
import json
import os
from datetime import datetime

import pandas as pd

from hoaxhertz.utils.fusion import FusionConfig, FusionModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV with label and head score columns.",
    )
    ap.add_argument("--model-type", type=str, default="logreg", choices=["logreg", "linsvm"])
    ap.add_argument(
        "--calibration",
        type=str,
        default="platt",
        choices=["platt", "isotonic", "none"],
    )
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--class-weight", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="artifacts/models/fusion")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs("apps/artifacts/reports", exist_ok=True)

    df = pd.read_csv(args.csv)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    # pick feature columns automatically (everything except 'label' and id-like fields)
    ignore = {"label", "audio_id", "id", "file"}
    feat_cols = [c for c in df.columns if c not in ignore]
    if len(feat_cols) == 0:
        raise ValueError("No head score columns found.")

    X = df[feat_cols].astype(float).values
    y = df["label"].astype(int).values

    cfg = FusionConfig(
        model_type=args.model_type,
        calibration=args.calibration,
        C=args.C,
        class_weight=None if args.class_weight in (None, "None") else args.class_weight,
        feature_order=feat_cols,
    )
    fm = FusionModel(cfg).fit(X, y, feature_order=feat_cols)

    # save model
    model_path = os.path.join(args.outdir, "model.joblib")
    fm.save(model_path)

    # write meta/report
    meta = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "csv": os.path.abspath(args.csv),
        "feature_order": feat_cols,
        "cfg": cfg.model_dump(),
        "metrics_cv": fm.artifacts.metrics_cv,
        "coef": None if fm.artifacts.coef_ is None else fm.artifacts.coef_.tolist(),
        "intercept": fm.artifacts.intercept_,
    }
    with open(os.path.join(args.outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # markdown report
    lines = []
    lines.append("# Fusion Report")
    lines.append("")
    lines.append(f"**Generated:** {meta['generated']}")
    lines.append(f"**Model:** {args.model_type} + calibration={args.calibration}")
    lines.append(f"**Features:** {', '.join(feat_cols)}")
    lines.append("")
    lines.append("## CV metrics")
    for k, v in fm.artifacts.metrics_cv.items():
        lines.append(f"- **{k}**: `{v:.4f}`")
    lines.append("")
    if fm.artifacts.coef_ is not None:
        lines.append("## Coefficients (logistic, interpretable)")
        for name, w in zip(feat_cols, fm.artifacts.coef_, strict=False):
            lines.append(f"- **{name}**: `{w:.4f}`")
        lines.append(f"- **intercept**: `{fm.artifacts.intercept_:.4f}`")
    lines.append("")
    lines.append(f"**Saved model:** `{model_path}`")
    rep_path = "apps/artifacts/reports/fusion_report.md"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Saved:", model_path)
    print("Report:", rep_path)


if __name__ == "__main__":
    main()
