#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates a consolidated HTML report for evaluations in:
  ml/factory/experiments/run_auto/eval/
Reads:
  - *_metrics.json (output from predict_batch.py)
  - *_preds.csv    (output from predict_batch.py)

Output:
  - report.html + imagens PNG (cm/roc) em _figs/

Requisitos: pandas, numpy, matplotlib, scikit-learn
"""

from __future__ import annotations
import argparse, json, base64, io, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report

# ---------- utils ----------
def _im_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

def _nice_name(tag: str) -> str:
    # exemplos de tag: svm_dev, logreg_train
    return (tag.replace("_", " "))
    
def _detect_dataset(path: str) -> str:
    p = path.lower()
    for name in ["partialspoof", "add", "had"]:
        if name in p:
            return {"partialspoof":"PartialSpoof","add":"ADD","had":"HAD"}[name]
    # fallback: use the first directory after "raw/" or "data/"
    m = re.search(r"/(raw|data)/([^/]+)/", p)
    if m:
        return m.group(2)
    return "unknown"

def _load_metrics_and_preds(eval_dir: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    # maps by file prefix (svm_dev, logreg_train, etc.)
    metrics_files = sorted(eval_dir.glob("*_metrics.json"))
    preds_files   = sorted(eval_dir.glob("*_preds.csv"))
    idx = {}
    for f in metrics_files:
        tag = f.name.replace("_metrics.json", "")
        out[tag] = {"metrics_path": f, "preds_path": None}
        out[tag]["metrics"] = json.loads(f.read_text(encoding="utf-8"))
    for f in preds_files:
        tag = f.name.replace("_preds.csv", "")
        out.setdefault(tag, {})
        out[tag]["preds_path"] = f
    return out

def _plot_confusion(y_true: List[str], y_pred: List[str], labels: List[str], title: str) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center")
    ax.set_xlabel("Pred"); ax.set_ylabel("True")
    fig.tight_layout()
    return _im_to_base64(fig)

def _plot_roc(y_true: List[str], probs: np.ndarray, labels: List[str], pos_label: str, title: str) -> Optional[str]:
    if len(labels) != 2:
        return None
    if pos_label not in labels:
        pos_label = labels[0]
    pos_idx = labels.index(pos_label)
    y_true_bin = np.array([1 if t == pos_label else 0 for t in y_true], dtype=np.int32)
    try:
        score = probs[:, pos_idx]
    except Exception:
        # sem probs? aborta
        return None
    try:
        fpr, tpr, _ = roc_curve(y_true_bin, score)
        auc = roc_auc_score(y_true_bin, score)
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}")
    ax.plot([0,1],[0,1], linestyle="--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _im_to_base64(fig)

def _safe(v, default=None):
    return v if v is not None else default

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=str, default="ml/factory/experiments/run_auto/eval")
    ap.add_argument("--out-html", type=str, default="ml/factory/experiments/run_auto/eval/report.html")
    ap.add_argument("--pos-class", type=str, default="fake")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise SystemExit(f"Eval dir not found: {eval_dir}")

    bundles = _load_metrics_and_preds(eval_dir)
    if not bundles:
        raise SystemExit("No *_metrics.json found.")

    # HTML building blocks
    sections = []
    summary_rows = []

    for tag, obj in sorted(bundles.items()):
        metrics = obj.get("metrics", {})
        preds_p = obj.get("preds_path", None)

        label_names = metrics.get("label_names") or metrics.get("classes") or ["fake", "real"]
        acc = metrics.get("accuracy", None)
        auc = None
        roc_auc_info = metrics.get("roc_auc_pos_class")
        if isinstance(roc_auc_info, dict):
            auc = roc_auc_info.get("roc_auc", None)

        # --- try to load preds for plots + breakdown by dataset ---
        y_true = y_pred = None
        probs = None
        cm_png = roc_png = None
        ds_table_html = ""
        head_table_html = ""
        n_used = metrics.get("n_used", None)

        if preds_p and preds_p.exists():
            df = pd.read_csv(preds_p)
            # Esperado: [path, true_label, pred_label, prob_fake, prob_real]
            # fallback: tenta detectar colunas
            cols = df.columns.str.lower().tolist()
            def find(colname):
                for i,c in enumerate(cols):
                    if c == colname: return df.columns[i]
                return None
            c_path = find("path")
            c_true = find("true_label")
            c_pred = find("pred_label")

            y_true = df[c_true].astype(str).tolist()
            y_pred = df[c_pred].astype(str).tolist()

            # try to build confusion matrix
            cm_png = _plot_confusion(y_true, y_pred, labels=label_names, title=f"Confusion • {_nice_name(tag)}")

            # try to extract probs in label_names order
            prob_cols = []
            for name in label_names:
                c = find(f"prob_{name.lower()}")
                if c: prob_cols.append(c)
            if len(prob_cols) == len(label_names):
                probs = df[prob_cols].to_numpy(dtype=np.float32)
                roc_png = _plot_roc(y_true, probs, label_names, args.pos_class, f"ROC • {_nice_name(tag)}")
            else:
                probs = None

            # ---------- breakdown por dataset ----------
            if c_path is not None:
                ds = df[c_path].astype(str).map(_detect_dataset)
                df2 = pd.DataFrame({
                    "dataset": ds,
                    "true": df[c_true].astype(str),
                    "pred": df[c_pred].astype(str)
                })
                # acurácia por dataset
                df2["ok"] = (df2["true"] == df2["pred"]).astype(np.int32)
                acc_by_ds = df2.groupby("dataset")["ok"].mean().sort_values(ascending=False)
                # suporte por dataset
                n_by_ds = df2["dataset"].value_counts()
                rows = []
                for d in acc_by_ds.index:
                    rows.append((d, float(acc_by_ds[d]), int(n_by_ds.get(d, 0))))
                # HTML
                ds_table_html = "<table class='t'><tr><th>Dataset</th><th>Accuracy</th><th>n</th></tr>" + \
                    "".join([f"<tr><td>{d}</td><td>{a:.4f}</td><td>{n}</td></tr>" for d,a,n in rows]) + \
                    "</table>"

            # cabeçalho rápido
            head_table_html = f"""
              <table class='t small'>
                <tr><th>Split/Model</th><th>Accuracy</th><th>ROC-AUC</th><th>n_used</th></tr>
                <tr><td>{_nice_name(tag)}</td><td>{_safe(acc,'-'):.4f}</td>
                    <td>{'-' if auc is None else f'{auc:.4f}'}</td>
                    <td>{_safe(n_used,'-')}</td></tr>
              </table>
            """

        # --- section block ---
        sec_html = f"""
          <h2>{_nice_name(tag)}</h2>
          {head_table_html}
          <div class='grid'>
            <div class='card'>
              <h3>Confusion Matrix</h3>
              {"<img src='data:image/png;base64," + cm_png + "'/>" if cm_png else "<p>—</p>"}
            </div>
            <div class='card'>
              <h3>ROC Curve</h3>
              {"<img src='data:image/png;base64," + roc_png + "'/>" if roc_png else "<p>—</p>"}
            </div>
          </div>
          <h3>Per-dataset breakdown</h3>
          {ds_table_html if ds_table_html else "<p>—</p>"}
          <hr/>
        """
        sections.append(sec_html)
        summary_rows.append((tag, acc, auc, n_used))

    # ---------- comparative summary ----------
    def tr(row):
        tag, acc, auc, n = row
        return f"<tr><td>{_nice_name(tag)}</td><td>{'-' if acc is None else f'{acc:.6f}'}</td><td>{'-' if auc is None else f'{auc:.6f}'}</td><td>{'-' if n is None else n}</td></tr>"
    summary_html = "<table class='t'><tr><th>Split / Model</th><th>Accuracy</th><th>ROC-AUC</th><th>n_used</th></tr>" + \
                   "".join([tr(r) for r in sorted(summary_rows)]) + "</table>"

    # ---------- final HTML ----------
    html = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Hoax-Hertz • Evaluation Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif; margin: 24px; color: #111; }}
h1 {{ margin: 0 0 12px 0; }}
h2 {{ margin: 24px 0 8px 0; }}
h3 {{ margin: 8px 0 8px 0; }}
hr {{ border: none; border-top: 1px solid #ddd; margin: 24px 0; }}
.t {{ border-collapse: collapse; width: 100%; margin: 8px 0 16px 0; }}
.t th, .t td {{ border: 1px solid #e6e6e6; padding: 6px 8px; text-align: left; }}
.t th {{ background: #fafafa; }}
.t.small td, .t.small th {{ padding: 4px 6px; font-size: 13px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
.card {{ border: 1px solid #eee; padding: 12px; border-radius: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
.note {{ color: #666; font-size: 13px; }}
.headerbox {{ display:flex; align-items:center; justify-content:space-between; gap:12px; }}
.kpi {{ display:flex; gap:16px; }}
.kpi .box {{ background:#fafafa; border:1px solid #eee; padding:8px 12px; border-radius:8px; }}
.mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
</style>
</head>
<body>
  <div class="headerbox">
    <h1>Hoax-Hertz • Evaluation Report</h1>
    <div class="kpi">
      <div class="box mono">dir: {eval_dir}</div>
      <div class="box mono">pos-class: {args.pos_class}</div>
    </div>
  </div>
  <p class="note">Relatório consolidado a partir de *_metrics.json e *_preds.csv. Imagens embutidas (base64).</p>

  <h2>Resumo</h2>
  {summary_html}

  {"".join(sections)}
</body>
</html>
"""

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"[OK] wrote HTML report: {out_html}")

if __name__ == "__main__":
    main()
