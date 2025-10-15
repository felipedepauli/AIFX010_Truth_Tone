#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build unified manifests for ADD (PF/RL), PartialSpoof, and HAD.

For EACH dataset root provided, create:
  <root>/_manifests/train_clips.csv      (path,label)
  <root>/_manifests/train_segments.csv   (path,start_sec,end_sec)   # only FAKE segments
  (if available) <root>/_manifests/dev_clips.csv, dev_segments.csv

Also (optional) combine all produced manifests into one folder:
  --combine-out ml/data/processed

This script is very tolerant:
- Scans label/ and protocols/ for CSV/TSV/TXT/JSON.
- Accepts many column names (path/utt/fname/audio, label/class, segments/partial/manip_segments/...).
- Parses segments from "a-b;c-d", pairs "a b c d", or JSON lists.
- Auto-detects milliseconds (if values look like ms, divides by 1000).

Usage examples:
  # Only ADD
  python tools/build_manifests.py --add-root ml/data/raw/ADD/ADD_train_dev

  # ADD + PartialSpoof + HAD and also write combined CSVs
  python tools/build_manifests.py \
      --add-root          ml/data/raw/ADD/ADD_train_dev \
      --partialspoof-root ml/data/raw/PartialSpoof \
      --had-root          ml/data/raw/HAD \
      --combine-out       ml/data/processed
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
# candidate field names
PATH_KEYS   = {"path","utt","utterance","fname","file","audio","relpath"}
LABEL_KEYS  = {"label","class","y","target","spoof_type","bonafide_spoof"}
SEG_KEYS    = {"segments","manip_segments","spoof_segments","partial","spoof_region","regions","boundaries"}
START_KEYS  = {"start","start_sec","seg_start","onset","t0"}
END_KEYS    = {"end","end_sec","seg_end","offset","t1"}

NUM = r"(?:\d+(?:\.\d+)?)"
RE_INLINE_SPANS = re.compile(rf"{NUM}\s*[-:]\s*{NUM}")  # "a-b" or "a:b"
RE_NUM          = re.compile(rf"^{NUM}$")

def _norm_label(x: str) -> str:
    x = (x or "").strip().lower()
    if any(w in x for w in ["bonafide","bona-fide","genuine","real"]): return "real"
    if any(w in x for w in ["spoof","fake","manip","partial","tampered"]): return "fake"
    return "fake" if ("spoof" in x or "fake" in x) else "real"

def _write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    rows_uniq = _uniq(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows_uniq)
    print(f"[OK] wrote {path} (n={len(rows_uniq)})")

def _uniq(rows: Iterable[List[str]]) -> List[List[str]]:
    seen=set(); out=[]
    for r in rows:
        t=tuple(r)
        if t not in seen: seen.add(t); out.append(r)
    return out

def _index_audio(root: Path, subdirs: List[str]) -> Dict[str, List[Path]]:
    idx: Dict[str, List[Path]] = {}
    for sub in subdirs:
        base = root / sub
        if not base.exists(): continue
        for p in base.rglob("*"):
            if p.suffix.lower() in AUDIO_EXTS:
                idx.setdefault(p.name.lower(), []).append(p)
                idx.setdefault(p.stem.lower(), []).append(p)
    return idx

def _find_audio(token: str, subset_base: Path, index: Dict[str, List[Path]]) -> Optional[Path]:
    # absolute/relative path-like?
    if any(sep in token for sep in ("/","\\")):
        p = (subset_base / token).resolve()
        if p.exists(): return p
        p2 = (subset_base.parent / token).resolve()
        if p2.exists(): return p2
    # id / basename / stem
    k = token.strip().lower()
    if k in index and index[k]: return index[k][0]
    if not k.endswith(".wav"):
        if (k+".wav") in index and index[k+".wav"]:
            return index[k+".wav"][0]
    return None

def _looks_like_ms(a: float, b: float) -> bool:
    # heurística: valores muito grandes e/ou com múltiplos trechos sugerem ms
    # exemplo: 120000 130000 (2 minutos)
    return (a>2000 and b>2000) or (b-a>5000)

def _parse_segments_tokens(parts: List[str]) -> List[Tuple[float,float]]:
    """Accept: 'a-b;c-d', or pairs: a b c d, or 'a:b' style."""
    segs: List[Tuple[float,float]] = []
    # inline spans
    inline = ";".join(parts)
    for m in RE_INLINE_SPANS.findall(inline):
        a,b = re.split(r"[-:]", m.replace(" ", ""))
        try:
            aa, bb = float(a), float(b)
            if bb < aa: aa, bb = bb, aa
            segs.append((aa, bb))
        except: pass
    if segs: return _maybe_ms_to_sec(segs)
    # numeric pairs
    nums = [float(x) for x in parts if RE_NUM.match(x)]
    for i in range(0, len(nums)-1, 2):
        a,b = nums[i], nums[i+1]
        if b < a: a,b = b,a
        segs.append((a,b))
    return _maybe_ms_to_sec(segs)

def _maybe_ms_to_sec(segs: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    if not segs: return segs
    ms_votes=0
    for a,b in segs:
        if _looks_like_ms(a,b): ms_votes+=1
    if ms_votes >= max(1, len(segs)//2):
        return [(a/1000.0, b/1000.0) for a,b in segs]
    return segs

# ---------- table readers (CSV/TSV/JSON/TXT) ----------
def _load_table_csv_tsv(path: Path) -> List[Dict[str,str]]:
    try:
        import pandas as pd  # optional
        df = pd.read_csv(path, sep="," if path.suffix==".csv" else "\t")
        return df.to_dict(orient="records")
    except Exception:
        rows=[]
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f) if path.suffix==".csv" else csv.reader(f, delimiter="\t")
            hdr = next(reader, [])
            for row in reader:
                rows.append({hdr[i]: row[i] if i<len(row) else "" for i in range(len(hdr))})
        return rows

def _parse_json_file(path: Path) -> List[Dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # unify to list
            data = data.get("items") or data.get("data") or data.get("entries") or [data]
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

def _extract_row_segments(row: Dict[str, str]) -> List[Tuple[float,float]]:
    """Try multiple keys/patterns to find segments in a tabular row."""
    # 1) single string field "segments"
    for k in SEG_KEYS:
        if k in row and isinstance(row[k], str) and row[k].strip():
            parts = [x for x in re.split(r"[;|]", row[k]) if x.strip()]
            cand = _parse_segments_tokens(parts)
            if cand: return cand
    # 2) start/end numeric columns
    s_key = next((k for k in START_KEYS if k in row), None)
    e_key = next((k for k in END_KEYS   if k in row), None)
    if s_key and e_key:
        try:
            a = float(row[s_key]); b = float(row[e_key])
            return _maybe_ms_to_sec([(min(a,b), max(a,b))])
        except Exception:
            pass
    # 3) JSON-like in a cell
    for k in SEG_KEYS:
        v = row.get(k, "")
        if isinstance(v, str) and "[" in v and "]" in v:
            try:
                arr = json.loads(v)
                out=[]
                for it in arr:
                    if isinstance(it, (list,tuple)) and len(it)>=2:
                        out.append((float(it[0]), float(it[1])))
                    elif isinstance(it, dict) and "start" in it and "end" in it:
                        out.append((float(it["start"]), float(it["end"])))
                if out: return _maybe_ms_to_sec(out)
            except Exception:
                pass
    return []

def _extract_row_path_label(row: Dict[str,str]) -> Tuple[str,str]:
    # path/id
    p_token = None
    for k in PATH_KEYS:
        if k in row:
            p_token = str(row[k]); break
    if p_token is None:
        # use first key as fallback
        p_token = next(iter(row.values()), "")
    # label
    lab = None
    for k in LABEL_KEYS:
        if k in row:
            lab = _norm_label(str(row[k])); break
    if lab is None:
        # inspect text
        lab = _norm_label(" ".join([str(row.get(k,"")) for k in row.keys()]))
    return p_token, lab

# ---------- ADD ----------
def build_add(add_root: Path) -> Dict[str, Path]:
    root = add_root
    if not ((root / "train").exists() and (root / "dev").exists() and (root / "label").exists()):
        if (add_root / "ADD_train_dev").exists():
            root = add_root / "ADD_train_dev"
    lbl_dir = root / "label"
    if not lbl_dir.exists():
        raise SystemExit(f"[ERR] ADD: label/ not found under {root}")

    out_dir = root / "_manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    index = _index_audio(root, ["train","dev"])

    def _consume_label_txt(txt: Path, subset: str) -> Tuple[List[List[str]], List[List[str]]]:
        clips: List[List[str]]=[]; segs: List[List[str]]=[]
        base = root / subset
        with txt.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"): continue
                parts = [p.strip() for p in (line.split(",") if "," in line else line.split()) if p.strip()]
                if len(parts)<2: continue
                token = parts[0]; lab = _norm_label(parts[1]); rest = parts[2:]
                p = _find_audio(token, base, index)
                if p is None:
                    p = _find_audio(parts[-1], base, index)
                if p is None:
                    print(f"[WARN] ADD: could not resolve path for: {line[:120]} ..."); continue
                clips.append([str(p), lab])
                if lab=="fake":
                    for a,b in _parse_segments_tokens(rest):
                        segs.append([str(p), a, b])
        return clips, segs

    # 1) official train/dev txt if present
    paths: Dict[str, Path] = {}
    tr_txt = lbl_dir / "train_label.txt"
    dv_txt = lbl_dir / "dev_label.txt"
    clips_all_tr: List[List[str]]=[]; segs_all_tr: List[List[str]]=[]
    clips_all_dv: List[List[str]]=[]; segs_all_dv: List[List[str]]=[]

    if tr_txt.exists():
        c,s = _consume_label_txt(tr_txt, "train"); clips_all_tr += c; segs_all_tr += s
    if dv_txt.exists():
        c,s = _consume_label_txt(dv_txt, "dev"); clips_all_dv += c; segs_all_dv += s

    # 2) any extra protocols under label/ or protocols/
    def _scan_misc(proto_root: Path, subset_guess: str):
        for path in proto_root.rglob("*"):
            if not path.is_file(): continue
            if path.suffix.lower() in {".csv",".tsv"}:
                rows = _load_table_csv_tsv(path)
                for r in rows:
                    p_token, lab = _extract_row_path_label(r)
                    # try both subsets if unsure
                    for subset in (subset_guess, "train","dev"):
                        p = _find_audio(p_token, root/subset, index)
                        if p is None: continue
                        if subset=="train": clips_all_tr.append([str(p), lab])
                        else:               clips_all_dv.append([str(p), lab])
                        if lab=="fake":
                            segs = _extract_row_segments(r)
                            for a,b in segs:
                                (segs_all_tr if subset=="train" else segs_all_dv).append([str(p), a, b])
                        break
            elif path.suffix.lower() == ".json":
                items = _parse_json_file(path)
                for it in items:
                    # path
                    p_token = None
                    for k in PATH_KEYS:
                        if k in it: p_token = str(it[k]); break
                    if p_token is None:
                        p_token = str(it.get("id", it.get("utt", "")))
                    # label
                    lab = None
                    for k in LABEL_KEYS:
                        if k in it: lab = _norm_label(str(it[k])); break
                    if lab is None:
                        lab = _norm_label(json.dumps(it))
                    # subset heuristics
                    for subset in (subset_guess,"train","dev"):
                        p = _find_audio(p_token, root/subset, index)
                        if p is None: continue
                        if subset=="train": clips_all_tr.append([str(p), lab])
                        else:               clips_all_dv.append([str(p), lab])
                        # segments
                        segs=[]
                        for k in SEG_KEYS:
                            if k in it:
                                val = it[k]
                                if isinstance(val, str):
                                    segs = _parse_segments_tokens([val])
                                elif isinstance(val, list):
                                    tmp=[]
                                    for x in val:
                                        if isinstance(x,(list,tuple)) and len(x)>=2:
                                            tmp.append((float(x[0]), float(x[1])))
                                        elif isinstance(x,dict) and "start" in x and "end" in x:
                                            tmp.append((float(x["start"]), float(x["end"])))
                                    segs = _maybe_ms_to_sec(tmp)
                                break
                        for a,b in segs:
                            (segs_all_tr if subset=="train" else segs_all_dv).append([str(p), a, b])
                        break
            elif path.suffix.lower() in {".txt",".lst"} and path.name not in {"train_label.txt","dev_label.txt"}:
                # generic TXT: try "utt label a-b;c-d"
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line=line.strip()
                        if not line or line.startswith("#"): continue
                        parts = [p.strip() for p in (line.split(",") if "," in line else line.split()) if p.strip()]
                        if len(parts)<2: continue
                        token=parts[0]; lab=_norm_label(parts[1]); rest=parts[2:]
                        found=False
                        for subset in (subset_guess,"train","dev"):
                            p = _find_audio(token, root/subset, index)
                            if p is None: 
                                p = _find_audio(parts[-1], root/subset, index)
                            if p is None: 
                                continue
                            found=True
                            if subset=="train": clips_all_tr.append([str(p), lab])
                            else:               clips_all_dv.append([str(p), lab])
                            if lab=="fake":
                                for a,b in _parse_segments_tokens(rest):
                                    (segs_all_tr if subset=="train" else segs_all_dv).append([str(p), a, b])
                            break
                        if not found:
                            # couldn't resolve — ignore silently
                            pass

    if (root/"label").exists():     _scan_misc(root/"label",     "train")
    if (root/"protocols").exists(): _scan_misc(root/"protocols", "train")

    # write
    _write_csv(out_dir/"train_clips.csv", ["path","label"], clips_all_tr)
    _write_csv(out_dir/"train_segments.csv", ["path","start_sec","end_sec"], segs_all_tr)
    if clips_all_dv:
        _write_csv(out_dir/"dev_clips.csv", ["path","label"], clips_all_dv)
    if segs_all_dv:
        _write_csv(out_dir/"dev_segments.csv", ["path","start_sec","end_sec"], segs_all_dv)

    return {
        "train_clips":   out_dir/"train_clips.csv",
        "train_segments":out_dir/"train_segments.csv",
        "dev_clips":     out_dir/"dev_clips.csv" if (out_dir/"dev_clips.csv").exists() else None,
        "dev_segments":  out_dir/"dev_segments.csv" if (out_dir/"dev_segments.csv").exists() else None,
    }

# ---------- PartialSpoof ----------
def build_partialspoof(ps_root: Path) -> Dict[str, Path]:
    out_dir = ps_root / "_manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_tr: List[List[str]]=[]; segs_tr: List[List[str]]=[]
    clips_dv: List[List[str]]=[]; segs_dv: List[List[str]]=[]

    def _consume_table_file(path: Path, split_hint: str):
        rows = _load_table_csv_tsv(path) if path.suffix.lower() in {".csv",".tsv"} else []
        # TXT/JSON
        if path.suffix.lower() == ".json":
            rows = _parse_json_file(path)
        elif path.suffix.lower() in {".txt",".lst"}:
            rows=[]
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line or line.startswith("#"): continue
                    parts=[p.strip() for p in (line.split(",") if "," in line else line.split()) if p.strip()]
                    if len(parts)<2: continue
                    rows.append({"path":parts[0], "label":parts[1], "segments":";".join(parts[2:]) if len(parts)>2 else ""})

        # base dirs
        base_tr = ps_root/"train" if (ps_root/"train").exists() else ps_root
        base_dv = ps_root/"dev"   if (ps_root/"dev").exists()   else ps_root
        idx = _index_audio(ps_root, ["train","dev","."])

        for r in rows:
            # path + label
            p_token, lab = _extract_row_path_label(r)
            # decide split
            # heuristic: prefer split_hint; fallback by existence under train/dev
            target = "train"
            p = _find_audio(p_token, base_tr, idx)
            if p is None:
                p = _find_audio(p_token, base_dv, idx)
                if p is not None: target="dev"
            elif split_hint=="dev":
                # if we were hinted dev, try moving
                p2 = _find_audio(p_token, base_dv, idx)
                if p2 is not None: p = p2; target="dev"
            if p is None:
                # last resort: treat token as full path relative to ps_root
                cand = (ps_root/p_token).resolve()
                if cand.exists(): p=cand
            if p is None: 
                # skip unresolved
                continue

            segs = _extract_row_segments(r)
            if target=="train":
                clips_tr.append([str(p), lab])
                if lab=="fake":
                    for a,b in segs: segs_tr.append([str(p), a, b])
            else:
                clips_dv.append([str(p), lab])
                if lab=="fake":
                    for a,b in segs: segs_dv.append([str(p), a, b])

    # scan protocols/
    protodir = ps_root/"protocols"
    if protodir.exists():
        for f in protodir.rglob("*"):
            if f.suffix.lower() in {".csv",".tsv",".txt",".json"}:
                _consume_table_file(f, "train")

    # fallback (no protocols found)
    if not clips_tr and not clips_dv:
        print("[WARN] PartialSpoof: no protocol files. Fallback by folder labels.")
        for split, base in [("train", ps_root/"train"), ("dev", ps_root/"dev")]:
            if not base.exists(): continue
            for wav in base.rglob("*"):
                if wav.suffix.lower() in AUDIO_EXTS:
                    lab = "fake" if any(s in wav.as_posix().lower() for s in ["/fake/","spoof"]) else "real"
                    (clips_tr if split=="train" else clips_dv).append([str(wav.resolve()), lab])

    # write
    _write_csv(out_dir/"train_clips.csv", ["path","label"], clips_tr)
    _write_csv(out_dir/"train_segments.csv", ["path","start_sec","end_sec"], segs_tr)
    if clips_dv:
        _write_csv(out_dir/"dev_clips.csv", ["path","label"], clips_dv)
    if segs_dv:
        _write_csv(out_dir/"dev_segments.csv", ["path","start_sec","end_sec"], segs_dv)

    return {
        "train_clips":   out_dir/"train_clips.csv",
        "train_segments":out_dir/"train_segments.csv",
        "dev_clips":     out_dir/"dev_clips.csv" if (out_dir/"dev_clips.csv").exists() else None,
        "dev_segments":  out_dir/"dev_segments.csv" if (out_dir/"dev_segments.csv").exists() else None,
    }

# ---------- HAD ----------
def build_had(had_root: Path) -> Dict[str, Path]:
    out_dir = had_root / "_manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips: List[List[str]]=[]; segs: List[List[str]]=[]

    # priority: CSV/TSV in root/protocols
    files = list(had_root.rglob("*.csv")) + list(had_root.rglob("*.tsv"))
    if not files:
        # TXT (utt label segments)
        files = list(had_root.rglob("*.txt"))

    idx = _index_audio(had_root, ["train","dev","."])

    for f in files:
        if f.suffix.lower() in {".csv",".tsv"}:
            rows = _load_table_csv_tsv(f)
            for r in rows:
                token, lab = _extract_row_path_label(r)
                # try resolve under train/dev/ or root
                p = _find_audio(token, had_root/"train", idx) or _find_audio(token, had_root/"dev", idx) or _find_audio(token, had_root, idx)
                if p is None: 
                    cand=(had_root/token).resolve()
                    if cand.exists(): p=cand
                if p is None: 
                    continue
                clips.append([str(p), lab])
                if lab=="fake":
                    for a,b in _extract_row_segments(r):
                        segs.append([str(p), a, b])

        elif f.suffix.lower() == ".txt":
            with f.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line=line.strip()
                    if not line or line.startswith("#"): continue
                    parts = [p.strip() for p in (line.split(",") if "," in line else line.split()) if p.strip()]
                    if len(parts)<2: continue
                    token, lab, rest = parts[0], _norm_label(parts[1]), parts[2:]
                    p = _find_audio(token, had_root, idx) or _find_audio(parts[-1], had_root, idx)
                    if p is None: continue
                    clips.append([str(p), lab])
                    if lab=="fake":
                        for a,b in _parse_segments_tokens(rest):
                            segs.append([str(p), a, b])

    _write_csv(out_dir/"train_clips.csv", ["path","label"], clips)
    _write_csv(out_dir/"train_segments.csv", ["path","start_sec","end_sec"], segs)
    return {
        "train_clips":   out_dir/"train_clips.csv",
        "train_segments":out_dir/"train_segments.csv"
    }

# ---------- combine ----------
def _merge_csvs(csv_paths: List[Optional[Path]], out_csv: Path, header: List[str]) -> None:
    rows=[]
    for p in csv_paths:
        if not p or not p.exists(): continue
        with p.open("r", encoding="utf-8") as f:
            rdr = csv.reader(f)
            hdr = next(rdr, [])
            for row in rdr:
                if len(row)>=len(header):
                    rows.append(row[:len(header)])
    _write_csv(out_csv, header, rows)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-root", type=str, default=None, help="ADD_train_dev or its parent")
    ap.add_argument("--partialspoof-root", type=str, default=None)
    ap.add_argument("--had-root", type=str, default=None)
    ap.add_argument("--combine-out", type=str, default=None, help="optional dir to write merged manifests")
    args = ap.parse_args()

    produced: Dict[str, List[Optional[Path]]] = {"train_clips":[], "train_segments":[], "dev_clips":[], "dev_segments":[]}

    if args.add_root:
        paths = build_add(Path(args.add_root).resolve())
        for k,v in paths.items():
            if v: produced[k].append(v)

    if args.partialspoof_root:
        paths = build_partialspoof(Path(args.partialspoof_root).resolve())
        for k,v in paths.items():
            if v: produced[k].append(v)

    if args.had_root:
        paths = build_had(Path(args.had_root).resolve())
        for k,v in paths.items():
            if v: produced[k].append(v)

    if args.combine_out:
        out = Path(args.combine_out).resolve(); out.mkdir(parents=True, exist_ok=True)
        if produced["train_clips"]:
            _merge_csvs(produced["train_clips"], out/"train_clips.csv", ["path","label"])
        if produced["train_segments"]:
            _merge_csvs(produced["train_segments"], out/"train_segments.csv", ["path","start_sec","end_sec"])
        if produced["dev_clips"]:
            _merge_csvs(produced["dev_clips"], out/"dev_clips.csv", ["path","label"])
        if produced["dev_segments"]:
            _merge_csvs(produced["dev_segments"], out/"dev_segments.csv", ["path","start_sec","end_sec"])

if __name__ == "__main__":
    main()
