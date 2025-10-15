#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferência de um único arquivo usando modelo treinado (joblib) e sidecar de features (*.feat.npz).

Uso:
  python ml/factory/pipeline/task_4_eval/predict_one.py \
    --audio /caminho/arquivo.wav \
    --model ml/factory/experiments/run0/clipclf.joblib \
    --suffix .feat.npz \
    --out-json artifacts/pred_one.json

Notas:
- O sidecar deve existir (gerado por task_1_preprocess/extract_features.py).
- Se o modelo foi treinado com cuML, o runtime precisa ter cuML/CuPy instalados.
- Para silenciar avisos de teardown do cuML/CuPy no encerramento do processo,
  usamos --fast-exit (padrão: ON), que finaliza com os._exit(0) no fim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any
import os
import sys

import numpy as np
import joblib


def _to_numpy(x):
    """Converte cuML/CuPy/pandas para numpy, se necessário."""
    try:
        import cupy as cp  # noqa
        if isinstance(x, cp.ndarray):
            return np.asarray(x.get())
    except Exception:
        pass
    try:
        import pandas as pd  # noqa
        if isinstance(x, (pd.Series, pd.DataFrame)):
            return x.to_numpy()
    except Exception:
        pass
    return np.asarray(x)


def _load_sidecar(sidecar: Path) -> dict:
    if not sidecar.exists():
        raise FileNotFoundError(f"sidecar não encontrado: {sidecar}")
    z = np.load(sidecar, allow_pickle=False)
    vec = z["vector"].astype(np.float32, copy=False)

    meta: Dict[str, Any] = {}
    # meta salvo como JSON string (np.array com .item()) ou ausente
    if "meta" in z.files:
        try:
            m = z["meta"]
            # pode vir como array de string (U/S) ou bytes
            if getattr(m, "dtype", None) is not None:
                if m.dtype.kind in {"U", "S"}:
                    meta = json.loads(m.item())
                elif m.dtype.kind == "V":  # structured
                    pass
                else:
                    # fallback: tenta decodificar bytes -> str -> json
                    s = m.tobytes().decode("utf-8", errors="ignore")
                    meta = json.loads(s)
        except Exception:
            meta = {}
    return {"vector": vec, "meta": meta}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    out = e / np.sum(e)
    return out.astype(np.float32, copy=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict single audio using joblib model + sidecar features")
    ap.add_argument("--audio", required=True, type=str, help="Caminho do áudio")
    ap.add_argument("--model", required=True, type=str, help="Caminho do .joblib treinado")
    ap.add_argument("--suffix", type=str, default=".feat.npz", help="Sufixo do sidecar (default: .feat.npz)")
    ap.add_argument("--out-json", type=str, default=None, help="Saída JSON (se ausente, imprime no stdout)")
    ap.add_argument("--fast-exit", dest="fast_exit", action="store_true", default=True,
                    help="Finaliza com os._exit(0) para evitar barulho de teardown do cuML/CuPy (default: ON)")
    ap.add_argument("--no-fast-exit", dest="fast_exit", action="store_false",
                    help="Desativa fast-exit (útil para debug)")
    args = ap.parse_args()

    audio = Path(args.audio)
    model_path = Path(args.model)
    sidecar = audio.with_suffix(audio.suffix + args.suffix)

    # 1) Carrega modelo
    obj = joblib.load(model_path)
    pipe = obj["model"]
    model_info: Dict[str, Any] = obj.get("model_info", {}) or {}

    # classes/label_names persistidas no treino (LabelEncoder)
    label_names = (
        model_info.get("label_names")
        or model_info.get("classes")
        or ["fake", "real"]
    )
    if not isinstance(label_names, (list, tuple)):
        label_names = list(label_names)
    vector_dim = int(model_info.get("vector_dim", 0))
    model_cfg_hash = str(model_info.get("cfg_hash", ""))

    # 2) Carrega sidecar
    payload = _load_sidecar(sidecar)
    x = payload["vector"]
    sidecar_meta = payload["meta"] or {}
    sidecar_hash = str(sidecar_meta.get("cfg_hash", ""))

    # sanity: dimensão e cfg_hash
    dim_ok = (vector_dim == 0) or (x.size == vector_dim)
    hash_ok = (not model_cfg_hash) or (model_cfg_hash == sidecar_hash)

    if not dim_ok:
        raise SystemExit(
            f"Dimensão incompatível: sidecar={x.size} vs modelo={vector_dim}. "
            f"Re-extraia features com a mesma config do treino."
        )
    if not hash_ok:
        print(f"[WARN] cfg_hash diverge: model={model_cfg_hash} vs sidecar={sidecar_hash} (seguindo mesmo assim)")

    # 3) Predição
    X = x.reshape(1, -1).astype(np.float32, copy=False)

    probs = None
    pred_idx = None

    # predict_proba (preferido)
    if hasattr(pipe, "predict_proba"):
        try:
            p = pipe.predict_proba(X)
            p = _to_numpy(p)
            probs = p[0] if p.ndim > 1 else p
        except Exception:
            probs = None

    # fallback com decision_function
    if probs is None:
        if hasattr(pipe, "decision_function"):
            s = pipe.decision_function(X)
            s = _to_numpy(s)
            s = s[0] if s.ndim > 1 else np.array([s], dtype=np.float32)
            if s.ndim == 1 and s.size == 1 and len(label_names) == 2:
                s = np.array([s[0], -s[0]], dtype=np.float32)
            probs = _softmax(s.astype(np.float32, copy=False))
        else:
            # último recurso
            ypred = pipe.predict(X)
            ypred = _to_numpy(ypred)
            pred_idx = int(ypred[0])
            probs = np.zeros((len(label_names),), dtype=np.float32)
            probs[pred_idx] = 1.0

    # índice previsto (se ainda não definido)
    if pred_idx is None:
        pred_idx = int(np.argmax(probs))

    # alinhar tamanho e construir dicionário de scores
    probs = np.asarray(probs, dtype=np.float32)
    probs = probs[: len(label_names)]
    scores = {label_names[i]: float(probs[i]) for i in range(len(probs))}
    pred_label = label_names[pred_idx]

    result = {
        "audio": str(audio),
        "model": str(model_path),
        "sidecar": str(sidecar),
        "pred_label": pred_label,
        "scores": scores,
        "top1": {"label": pred_label, "prob": float(probs[pred_idx])},
        "vector_dim": int(x.size),
        "cfg_hash_check": {
            "model": model_cfg_hash,
            "sidecar": sidecar_hash,
            "match": bool(hash_ok),
        },
    }

    js = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(js + "\n")
        print(f"[OK] saved: {args.out_json}")
    else:
        print(js)

    # --- liberar memórias (opcional) ---
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        import gc
        del pipe
        gc.collect()
    except Exception:
        pass

    # --- fast exit para silenciar teardown do cuML/CuPy ---
    if args.fast_exit:
        os._exit(0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
