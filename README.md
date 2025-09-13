## CS180 Project 1 — Prokudin-Gorskii Colorizer

# Overview
- This repo contains a Python implementation that splits the B/G/R plates from a single grayscale glass-plate scan and aligns G and R to B to create a color image.

# Quick Start (batch all images)
- From the repo root, run:
  - `python3 code/run_all_images.py`
- What it does:
  - TIFs: uses a pyramid + Sobel edges metric.
  - JPGs: uses single‑scale search with NCC.
- Outputs are written to `output/<basename>_color.jpg`.

# Running A Single Image
- Example, small JPG (single‑scale NCC):
  - `python3 code/proj1_colorizer.py --input data/cathedral.jpg --method single --metric ncc --max-radius 15 --auto-crop`
- Example, large TIF (pyramid + edges):
  - `python3 code/proj1_colorizer.py --input data/emir.tif --method pyramid --metric edges5 --auto-crop`
