import os, glob, subprocess, sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "output")
SCRIPT = os.path.join(os.path.dirname(__file__), "proj1_colorizer.py")
PY = sys.executable

def run(img_path, extra_args):
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(OUT_DIR, f"{base}_color.jpg")
    cmd = [PY, SCRIPT, "--input", img_path, "--output", out_path] + extra_args
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    # TIFs: pyramid + edges5
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.tif")) + glob.glob(os.path.join(DATA_DIR, "*.tiff"))):
        run(f, ["--method", "pyramid", "--metric", "edges5", "--auto-crop"])

    # JPGs: single-scale + NCC
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")) + glob.glob(os.path.join(DATA_DIR, "*.jpeg"))):
        run(f, ["--method", "single", "--metric", "ncc", "--max-radius", "15", "--auto-crop"])

if __name__ == "__main__":
    main()

