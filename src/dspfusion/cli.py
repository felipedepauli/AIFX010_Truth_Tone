import argparse, json, os
from pathlib import Path
import pandas as pd
from dspfusion.preproc.base import PreprocConfig, load_and_standardize
from dspfusion.features.mfcc import mfcc_features
from dspfusion.features.enf import estimate_enf
from dspfusion.features.ambient import ambient_signature
from dspfusion.detectors.locutor import speaker_change_scores
from dspfusion.detectors.enfdet import enf_discontinuity
from dspfusion.detectors.ambientdet import ambient_change
from dspfusion.utils.fusion import fuse_scores

def process_file(fp:str):
    cfg = PreprocConfig()
    y,sr = load_and_standardize(fp, cfg)
    M = mfcc_features(y,sr)
    spk = speaker_change_scores(M)

    fi = estimate_enf(y,sr)
    enf = enf_discontinuity(fi)

    amb = ambient_change([ambient_signature(y,sr)]*3)  # placeholder sequência

    fused = fuse_scores(spk, enf, amb)
    return {
        "file": fp,
        "spk_len": len(spk),
        "enf_len": len(enf),
        "amb_len": len(amb),
        "fused_len": 0 if fused is None else len(fused),
        "score_mean": None if fused is None else float(fused.mean())
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="pasta com WAV/FLAC")
    ap.add_argument("--out", required=True, help="CSV de saída")
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    files = [str(p) for p in Path(args.input).rglob("*.wav")]
    rows = [process_file(f) for f in files]
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[ok] relatório: {args.out}")

if __name__ == "__main__":
    main()
