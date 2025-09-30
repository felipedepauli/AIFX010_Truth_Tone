from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os
from dspfusion.preproc.base import PreprocConfig, load_and_standardize
from dspfusion.features.mfcc import mfcc_features
from dspfusion.features.enf import estimate_enf
from dspfusion.features.ambient import ambient_signature
from dspfusion.detectors.locutor import speaker_change_scores
from dspfusion.detectors.enfdet import enf_discontinuity
from dspfusion.detectors.ambientdet import ambient_change
from dspfusion.utils.fusion import fuse_scores

app = FastAPI(title="Audio Splicing (DSP/ML clássico)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read()); tmp_path = tmp.name
    try:
        cfg = PreprocConfig()
        y,sr = load_and_standardize(tmp_path, cfg)
        spk = speaker_change_scores(mfcc_features(y,sr))
        enf = enf_discontinuity(estimate_enf(y,sr))
        amb = ambient_change([ambient_signature(y,sr)]*3)
        fused = fuse_scores(spk,enf,amb)
        return {
            "spk_len": len(spk), "enf_len": len(enf), "amb_len": len(amb),
            "fused_mean": None if fused is None else float(fused.mean())
        }
    finally:
        os.unlink(tmp_path)
