from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
from typing import Dict, Any, List

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all images (TIF with pyramid+edges5, JPG with single+NCC).")
    p.add_argument("--data", default="data", help="Directory containing input images (default: data)")
    p.add_argument("--script", default="proj1_colorizer.py", help="Path to your colorizer script (default: proj1_colorizer.py)")
    p.add_argument("--outdir", default="output", help="Directory to save outputs (default: output)")
    p.add_argument("--jpg-max-radius", type=int, default=15, help="max radius for single-scale JPG runs (default: 15)")
    return p.parse_args()

def is_tif(path: str) -> bool:
    return path.lower().endswith((".tif", ".tiff"))

def is_jpg(path: str) -> bool:
    return path.lower().endswith((".jpg", ".jpeg"))

def run_one(python_exec: str, script: str, inp: str, out_path: str, args: List[str]) -> str:
    cmd = [python_exec, script, "--input", inp, "--output", out_path, *args]
    print(">", " ".join(map(str, cmd)))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return proc.stdout

# parser for script’s prints
RE_HEADER = re.compile(
    r"Method:\s*(?P<method>\S+),\s*Anchor:\s*(?P<anchor>\S+),\s*Metric:\s*(?P<metric>\S+),\s*Pre-crop:\s*(?P<pre>[0-9.]+),\s*Border:\s*(?P<border>[0-9.]+)"
)
RE_G_ALIGNED = re.compile(r"(Aligned|Applied shift to)\s*G\s*(to B|\(relative to B\)):\s*dy=([-+]?\d+),\s*dx=([-+]?\d+)")
RE_R_ALIGNED = re.compile(r"(Aligned|Applied shift to)\s*R\s*(to B|\(relative to B\)):\s*dy=([-+]?\d+),\s*dx=([-+]?\d+)")
RE_SAVED = re.compile(r"Saved:\s*(.+)")

def parse_report(stdout: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    m = RE_HEADER.search(stdout)
    if m:
        meta.update(
            method=m.group("method"),
            anchor=m.group("anchor"),
            metric=m.group("metric"),
            pre_crop=float(m.group("pre")),
            border=float(m.group("border")),
        )
    g = RE_G_ALIGNED.search(stdout)
    if g:
        meta["g"] = [int(g.group(3)), int(g.group(4))]
    r = RE_R_ALIGNED.search(stdout)
    if r:
        meta["r"] = [int(r.group(3)), int(r.group(4))]
    s = RE_SAVED.search(stdout)
    if s:
        meta["output_file"] = os.path.basename(s.group(1).strip())
    return meta

def main() -> None:
    args = parse_args()
    py = sys.executable
    os.makedirs(args.outdir, exist_ok=True)

    # Collect files
    tif_files = sorted(
        glob.glob(os.path.join(args.data, "*.tif")) + glob.glob(os.path.join(args.data, "*.tiff"))
    )
    jpg_files = sorted(
        glob.glob(os.path.join(args.data, "*.jpg")) + glob.glob(os.path.join(args.data, "*.jpeg"))
    )

    results: List[Dict[str, Any]] = []
    failures: List[str] = []

    # TIFs: pyramid + edges5
    for f in tif_files:
        base = os.path.splitext(os.path.basename(f))[0]
        out_img = os.path.join(args.outdir, f"{base}_color.jpg")
        try:
            stdout = run_one(
                py,
                args.script,
                f,
                out_img,
                ["--method", "pyramid", "--metric", "edges5", "--auto-crop"],
            )
            meta = parse_report(stdout)
            meta["input"] = f
            results.append(meta)
        except Exception as e:
            print(f"[ERROR] {f}: {e}")
            failures.append(f)

    # JPGs: single + ncc
    for f in jpg_files:
        base = os.path.splitext(os.path.basename(f))[0]
        out_img = os.path.join(args.outdir, f"{base}_color.jpg")
        try:
            stdout = run_one(
                py,
                args.script,
                f,
                out_img,
                ["--method", "single", "--metric", "ncc", "--max-radius", str(args.jpg_max_radius), "--auto-crop"],
            )
            meta = parse_report(stdout)
            meta["input"] = f
            results.append(meta)
        except Exception as e:
            print(f"[ERROR] {f}: {e}")
            failures.append(f)

    # Build offsets.json for the webpage
    offsets: Dict[str, Any] = {}
    for r in results:
        of = r.get("output_file")
        if not of or "g" not in r or "r" not in r:
            continue
        offsets[of] = {"g": r["g"], "r": r["r"]}

    with open(os.path.join(args.outdir, "offsets.json"), "w") as f:
        json.dump(offsets, f, indent=2)
    print(f"Wrote {os.path.join(args.outdir, 'offsets.json')}")

    # CSV summary
    with open(os.path.join(args.outdir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["output_file", "input", "method", "metric", "anchor", "pre_crop", "border", "g_dy", "g_dx", "r_dy", "r_dx"])
        for r in results:
            w.writerow([
                r.get("output_file", ""),
                r.get("input", ""),
                r.get("method", ""),
                r.get("metric", ""),
                r.get("anchor", ""),
                r.get("pre_crop", ""),
                r.get("border", ""),
                *(r.get("g", ["", ""])),
                *(r.get("r", ["", ""])),
            ])
    print(f"Wrote {os.path.join(args.outdir, 'summary.csv')}")

    if failures:
        print("\nSome files failed:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    else:
        print("\nAll done ✅")

if __name__ == "__main__":
    main()
