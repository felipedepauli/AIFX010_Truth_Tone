#!/usr/bin/env python3
"""
Download selected audio datasets (WAV-ready) with zero external CLIs.

Datasets supported:
- ESC50 (GitHub ZIP)
- UrbanSound8K (Zenodo TAR.GZ, with MD5 check)
- DEMAND (Zenodo: choose 16k zips)
- MUSAN (OpenSLR 17)
- RIR_NOISE_SLR28 (OpenSLR 28)
- TAU2022_MOBILE_DEV (Zenodo: 16 part zips + meta)

Usage examples
--------------
# baixar tudo no padrão ml/data/raw
python tools/download_datasets.py

# escolher datasets e dir
python tools/download_datasets.py --out ml/data/raw --datasets ESC50 UrbanSound8K DEMAND

# apenas TAU 2022 (atenção: ~27 GB)
python tools/download_datasets.py --datasets TAU2022_MOBILE_DEV
"""
from __future__ import annotations
import argparse, os, sys, hashlib, tarfile, zipfile, time
from pathlib import Path
from urllib.request import urlopen, Request

# ---------- tiny utils ----------
def _human(n: float) -> str:
    for u in ['B','KB','MB','GB','TB']:
        if n < 1024.0: return f"{n:,.1f} {u}"
        n /= 1024.0
    return f"{n:.1f} PB"

def _download(url: str, dst: Path, chunk=1024*1024) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r, open(dst, "wb") as f:
        total = r.length or 0
        read = 0
        t0 = time.time()
        while True:
            b = r.read(chunk)
            if not b: break
            f.write(b)
            read += len(b)
            if total:
                pct = 100.0 * read / total
                rate = read / max(1e-6, (time.time()-t0))
                print(f"\r[DL] {dst.name} {_human(read)} / {_human(total)} ({pct:5.1f}%) @ {_human(rate)}/s", end="")
        print()
    print(f"[OK] Downloaded: {dst}")

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def _extract(archive: Path, out_dir: Path) -> None:
    print(f"[EXTRACT] {archive.name} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffixes[-2:] == ['.tar', '.gz'] or archive.suffix == '.tgz':
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(out_dir)
    elif archive.suffix == '.zip':
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(out_dir)
    else:
        print(f"[WARN] Unknown archive type: {archive}")
    print(f"[OK] Extracted: {archive.name}")

# ---------- dataset recipes ----------
def esc50(base: Path) -> None:
    """
    ESC-50 official GitHub ZIP (≈600 MB)
    Source: https://github.com/karolpiczak/ESC-50  (Download zip)
    """
    url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    out = base / "ESC-50"
    zip_path = out / "ESC-50-master.zip"
    if not zip_path.exists():
        _download(url, zip_path)
    _extract(zip_path, out)
    print("[INFO] WAVs expected under ESC-50/ESC-50-master/audio")

def urbansound8k(base: Path) -> None:
    """
    UrbanSound8K TAR.GZ from Zenodo (≈6 GB)
    Record: https://zenodo.org/records/1203745
    """
    out = base / "UrbanSound8K"
    url = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1"
    tgz = out / "UrbanSound8K.tar.gz"
    md5_expected = "9aa69802bbf37fb986f71ec1483a196e"  # Zenodo
    if not tgz.exists():
        _download(url, tgz)
    md5_got = _md5(tgz)
    print(f"[CHECK] MD5 {md5_got}")
    if md5_got != md5_expected:
        print("[WARN] MD5 mismatch (source lists 9aa6980...). File may be corrupt or updated.")
    _extract(tgz, out)

def demand_16k(base: Path) -> None:
    """
    DEMAND (16k zips) from Zenodo record:
    https://zenodo.org/records/1227121
    """
    out = base / "DEMAND"
    # Only 16k variants to economize espaço
    names = [
        "DKITCHEN_16k.zip","DLIVING_16k.zip","DWASHING_16k.zip","NFIELD_16k.zip",
        "NPARK_16k.zip","NRIVER_16k.zip","OHALLWAY_16k.zip","OMEETING_16k.zip",
        "OOFFICE_16k.zip","PCAFETER_16k.zip","PRESTO_16k.zip","PSTATION_16k.zip",
        "SPSQUARE_16k.zip","STRAFFIC_16k.zip","TBUS_16k.zip","TCAR_16k.zip","TMETRO_16k.zip"
    ]
    base_url = "https://zenodo.org/records/1227121/files/"
    for name in names:
        url = f"{base_url}{name}?download=1"
        zpath = out / name
        if not zpath.exists():
            _download(url, zpath)
        _extract(zpath, out)

def musan(base: Path) -> None:
    """
    MUSAN (OpenSLR 17)
    Homepage: https://www.openslr.org/17/
    Direct:   https://www.openslr.org/resources/17/musan.tar.gz
    """
    out = base / "MUSAN"
    url = "https://www.openslr.org/resources/17/musan.tar.gz"
    tgz = out / "musan.tar.gz"
    if not tgz.exists():
        _download(url, tgz)
    _extract(tgz, out)

def rir_noise_slr28(base: Path) -> None:
    """
    Room Impulse Response and Noise Database (OpenSLR 28)
    Homepage: https://www.openslr.org/28/
    Direct:   https://www.openslr.org/resources/28/rirs_noises.zip
    """
    out = base / "SLR28_RIRS_NOISES"
    url = "https://www.openslr.org/resources/28/rirs_noises.zip"
    z = out / "rirs_noises.zip"
    if not z.exists():
        _download(url, z)
    _extract(z, out)

def tau2022_mobile_dev(base: Path) -> None:
    """
    TAU Urban Acoustic Scenes 2022 Mobile (development)
    Zenodo record (files list): https://zenodo.org/records/6337421
    """
    out = base / "TAU_urban_asc_2022_mobile_dev"
    # 16 audio parts + meta; ~27 GB total
    base_url = "https://zenodo.org/records/6337421/files/"
    parts = [f"TAU-urban-acoustic-scenes-2022-mobile-development.audio.{i}.zip" for i in range(1,17)]
    parts += ["TAU-urban-acoustic-scenes-2022-mobile-development.meta.zip"]
    for name in parts:
        url = f"{base_url}{name}?download=1"
        zpath = out / name
        if not zpath.exists():
            _download(url, zpath)
        _extract(zpath, out)

# ---------- main ----------
RECIPES = {
    # "ESC50": esc50,
    # "UrbanSound8K": urbansound8k,
    "DEMAND": demand_16k,
    # "MUSAN": musan,
    # "RIR_NOISE_SLR28": rir_noise_slr28,
    "TAU2022_MOBILE_DEV": tau2022_mobile_dev,
}

def main():
    ap = argparse.ArgumentParser(description="Download audio datasets (official sources)")
    ap.add_argument("--out", type=str, default="ml/data/raw", help="Output root directory")
    ap.add_argument("--datasets", nargs="+",
                    default=list(RECIPES.keys()),
                    choices=list(RECIPES.keys()),
                    help="Datasets to download")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output dir: {out_root}")

    for name in args.datasets:
        print(f"\n=== [{name}] ===")
        try:
            RECIPES[name](out_root)
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")

    print("\n[OK] All done.")

if __name__ == "__main__":
    main()
