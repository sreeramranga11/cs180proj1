import os, glob, subprocess, sys

DATA_DIR = "data"
OUT_DIR = "output"
SCRIPT = "proj1_colorizer.py"

PY = sys.executable
# uses current venv's python

def run(img_path, extra_args):
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(OUT_DIR, f"{base}_color.jpg")
    cmd = [PY, SCRIPT, "--input", img_path, "--output", out_path] + extra_args
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    # TIFs: pyramid + edges5
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.tif"))):
        run(f, ["--method", "pyramid", "--metric", "edges5", "--auto-crop"])

    # JPGs: single-scale + NCC
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg"))):
        run(f, ["--method", "single", "--metric", "ncc", "--max-radius", "15", "--auto-crop"])

if __name__ == "__main__":
    main()
