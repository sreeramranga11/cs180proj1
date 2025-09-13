import argparse
import os
from typing import Tuple

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rescale
from skimage.filters import sobel, gaussian
from skimage.util import img_as_float, img_as_ubyte
from skimage.registration import phase_cross_correlation


def crop_border(arr: np.ndarray, border: float | int) -> np.ndarray:
    """Crop a border from all sides. If border < 1, treat as fraction, else pixels."""
    h, w = arr.shape[:2]
    if border < 1:
        by = int(max(0, min(h // 2 - 1, round(border * h))))
        bx = int(max(0, min(w // 2 - 1, round(border * w))))
    else:
        by = int(min(h // 2 - 1, border))
        bx = int(min(w // 2 - 1, border))
    if by <= 0 and bx <= 0:
        return arr
    return arr[by : h - by, bx : w - bx]


def crop_border_xy(arr: np.ndarray, border_y: float | int, border_x: float | int) -> np.ndarray:
    """Crop possibly different vertical (border_y) and horizontal (border_x) borders.

    If a value < 1, it's treated as a fraction of the dimension; otherwise as pixels.
    """
    h, w = arr.shape[:2]
    # vertical
    if border_y < 1:
        by = int(max(0, min(h // 2 - 1, round(border_y * h))))
    else:
        by = int(min(h // 2 - 1, border_y))
    # horizontal
    if border_x < 1:
        bx = int(max(0, min(w // 2 - 1, round(border_x * w))))
    else:
        bx = int(min(w // 2 - 1, border_x))
    if by <= 0 and bx <= 0:
        return arr
    return arr[by : h - by, bx : w - bx]


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    k = max(3, int(k) | 1)
    ker = np.ones(k, dtype=float) / k
    return np.convolve(x, ker, mode="same")


def detect_content_bbox(
    img: np.ndarray,
    limit_frac: float = 0.25,
    inward: int = 1,
    debug_dir: str | None = None,
    tag: str | None = None,
) -> tuple[int, int, int, int]:
    """Detect content bounding box via Sobel edge energy near borders.

    Returns (top, bottom, left, right), where bottom/right are exclusive when slicing.
    """
    h, w = img.shape
    limit_r = max(1, int(round(limit_frac * h)))
    limit_c = max(1, int(round(limit_frac * w)))

    edges = sobel(img)
    row_e = edges.sum(axis=1)
    col_e = edges.sum(axis=0)
    row_e_s = _moving_average(row_e, max(5, (h // 100) * 2 + 1))
    col_e_s = _moving_average(col_e, max(5, (w // 100) * 2 + 1))

    def pick_from_start(s: np.ndarray, limit: int) -> int:
        window = s[:limit]
        if window.size == 0:
            return 0
        g = np.gradient(window)
        i1 = int(np.argmax(window))
        i2 = int(np.argmax(np.maximum(g, 0)))
        return max(0, min(i1, i2))

    def pick_from_end(s: np.ndarray, limit: int, total: int) -> int:
        window = s[::-1][:limit]
        if window.size == 0:
            return total - 1
        g = np.gradient(window)
        i1 = int(np.argmax(window))
        i2 = int(np.argmax(np.maximum(g, 0)))
        idx_from_end = max(0, min(i1, i2))
        return total - 1 - idx_from_end

    top = pick_from_start(row_e_s, limit_r) + inward
    bottom = pick_from_end(row_e_s, limit_r, h) - inward
    left = pick_from_start(col_e_s, limit_c) + inward
    right = pick_from_end(col_e_s, limit_c, w) - inward

    top = int(np.clip(top, 0, max(0, h - 2)))
    bottom = int(np.clip(bottom, top + 1, h))
    left = int(np.clip(left, 0, max(0, w - 2)))
    right = int(np.clip(right, left + 1, w))

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        base = tag or "img"
        e_norm = _normalize01(edges)
        imsave(os.path.join(debug_dir, f"{base}_edges.jpg"), img_as_ubyte(e_norm))
        mask = np.zeros_like(img)
        mask[top:bottom, left:right] = 1
        preview = e_norm * 0.5 + mask * 0.5
        imsave(os.path.join(debug_dir, f"{base}_edges_with_box.jpg"), img_as_ubyte(_normalize01(preview)))
        with open(os.path.join(debug_dir, f"{base}_row_energy.csv"), "w") as f:
            f.write("row,sum\n")
            for i, v in enumerate(row_e_s):
                f.write(f"{i},{float(v)}\n")
        with open(os.path.join(debug_dir, f"{base}_col_energy.csv"), "w") as f:
            f.write("col,sum\n")
            for j, v in enumerate(col_e_s):
                f.write(f"{j},{float(v)}\n")

    return top, bottom, left, right

def detect_black_bbox(
    img: np.ndarray,
    limit_frac: float = 0.35,
    inward: int = 0,
    debug_dir: str | None = None,
    tag: str | None = None,
) -> tuple[int, int, int, int]:
    """Detect inner edges of black borders via intensity profiles per side.

    Works best on glass-plate scans with dark frames. Returns (top, bottom, left, right)
    with bottom/right exclusive for slicing.
    """
    h, w = img.shape
    k_r = max(5, (h // 100) * 2 + 1)
    k_c = max(5, (w // 100) * 2 + 1)
    rprof = _moving_average(img.mean(axis=1), k_r)
    cprof = _moving_average(img.mean(axis=0), k_c)

    lim_r = max(1, int(round(limit_frac * h)))
    lim_c = max(1, int(round(limit_frac * w)))

    def from_start(p: np.ndarray, lim: int) -> int:
        win = p[:lim]
        if win.size == 0:
            return 0
        black = np.percentile(win, 20)
        inner = np.percentile(p[lim:] if p.size > lim else p, 70)
        thr = black + 0.45 * (inner - black)
        idx = int(np.argmax(win > thr)) if np.any(win > thr) else int(np.argmax(np.gradient(win)))
        return idx

    def from_end(p: np.ndarray, lim: int, total: int) -> int:
        win = p[::-1][:lim]
        if win.size == 0:
            return total - 1
        black = np.percentile(win, 20)
        inner = np.percentile(p[:-lim] if total > lim else p, 70)
        thr = black + 0.45 * (inner - black)
        idx_from_end = int(np.argmax(win > thr)) if np.any(win > thr) else int(np.argmax(np.gradient(win)))
        return total - 1 - idx_from_end

    top = from_start(rprof, lim_r) + inward
    bottom = from_end(rprof, lim_r, h) - inward
    left = from_start(cprof, lim_c) + inward
    right = from_end(cprof, lim_c, w) - inward

    top = int(np.clip(top, 0, max(0, h - 2)))
    bottom = int(np.clip(bottom, top + 1, h))
    left = int(np.clip(left, 0, max(0, w - 2)))
    right = int(np.clip(right, left + 1, w))

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        base = tag or "img"
        with open(os.path.join(debug_dir, f"{base}_row_profile.csv"), "w") as f:
            f.write("row,mean\n")
            for i, v in enumerate(rprof):
                f.write(f"{i},{float(v)}\n")
        with open(os.path.join(debug_dir, f"{base}_col_profile.csv"), "w") as f:
            f.write("col,mean\n")
            for j, v in enumerate(cprof):
                f.write(f"{j},{float(v)}\n")
        # Save overlay preview
        m = np.zeros_like(img)
        m[top:bottom, left:right] = 1
        preview = 0.6 * _normalize01(img) + 0.4 * m
        imsave(os.path.join(debug_dir, f"{base}_mean_with_box.jpg"), img_as_ubyte(_normalize01(preview)))

    return top, bottom, left, right


def _overlap_slices(h: int, w: int, dy: int, dx: int) -> Tuple[slice, slice, slice, slice]:
    """Return slices for overlapping region between two images of size h x w when the moving image is shifted by (dy, dx) relative to the reference.

    Returns: (ys_m, xs_m, ys_r, xs_r) slices to index into moving and reference arrays respectively.
    """
    # Vertical overlap
    if dy >= 0:
        ys_m = slice(dy, h)
        ys_r = slice(0, h - dy)
    else:
        ys_m = slice(0, h + dy)
        ys_r = slice(-dy, h)

    # Horizontal overlap
    if dx >= 0:
        xs_m = slice(dx, w)
        xs_r = slice(0, w - dx)
    else:
        xs_m = slice(0, w + dx)
        xs_r = slice(-dx, w)

    return ys_m, xs_m, ys_r, xs_r


def _prepare_for_metric(img: np.ndarray, metric: str) -> np.ndarray:
    """Preprocess the channel for a given metric to improve robustness.

    - 'edges' uses Sobel magnitude to be robust to brightness changes.
    - 'ssd' and 'ncc' use the raw image.
    """
    if metric == "edges":
        return sobel(img)
    if metric == "edges5":
        # Slightly blur first to capture broader edges (approx larger kernel)
        return sobel(gaussian(img, sigma=1.6, preserve_range=True))
    return img


def _metric_score(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    """Compute similarity score between two 2D arrays.

    For 'ssd' return negative SSD (so higher is better). For 'ncc' return NCC.
    For 'edges' default to negative SSD on Sobel magnitudes.
    """
    if a.size == 0 or b.size == 0:
        return -np.inf

    if metric == "ncc":
        a0 = a - a.mean()
        b0 = b - b.mean()
        denom = np.linalg.norm(a0) * np.linalg.norm(b0)
        if denom == 0:
            return -np.inf
        return float(np.sum(a0 * b0) / denom)

    # SSD (and edges via SSD)
    diff = a - b
    ssd = float(np.sum(diff * diff))
    # Return negative SSD to keep the convention: higher score is better
    return -ssd

def _score_at(moving: np.ndarray, reference: np.ndarray, dy: int, dx: int,
              metric: str, border: float | int = 0.1,
              border_x: float | int | None = None) -> float:
    """Score a given (dy,dx) using the SAME feature space as 'metric'."""
    h, w = reference.shape
    ys_m, xs_m, ys_r, xs_r = _overlap_slices(h, w, dy, dx)
    a = moving[ys_m, xs_m]
    b = reference[ys_r, xs_r]
    a = _prepare_for_metric(a, metric)
    b = _prepare_for_metric(b, metric)
    if border_x is None:
        a = crop_border(a, border)
        b = crop_border(b, border)
    else:
        a = crop_border_xy(a, border, border_x)
        b = crop_border_xy(b, border, border_x)
    # Evaluate all candidates on the same scale: NCC in their feature space
    return _metric_score(a, b, "ncc")

def exhaustive_search(
    moving: np.ndarray,
    reference: np.ndarray,
    dy_range: range,
    dx_range: range,
    metric: str = "ncc",
    border: float | int = 0.1,
    border_x: float | int | None = None,
) -> Tuple[int, int, float]:
    """Exhaustive integer-translation search over dy_range x dx_range.

    Returns (best_dy, best_dx, best_score).
    """
    h, w = reference.shape
    mov_proc = _prepare_for_metric(moving, metric)
    ref_proc = _prepare_for_metric(reference, metric)

    best_score = -np.inf
    best = (0, 0)

    for dy in dy_range:
        for dx in dx_range:
            ys_m, xs_m, ys_r, xs_r = _overlap_slices(h, w, dy, dx)
            a = mov_proc[ys_m, xs_m]
            b = ref_proc[ys_r, xs_r]

            # Crop borders inside overlapped region to avoid edge artifacts
            if border_x is None:
                a_c = crop_border(a, border)
                b_c = crop_border(b, border)
            else:
                a_c = crop_border_xy(a, border, border_x)
                b_c = crop_border_xy(b, border, border_x)
            score = _metric_score(a_c, b_c, metric if metric != "edges" else "ssd")
            if score > best_score:
                best_score = score
                best = (dy, dx)
    return best[0], best[1], best_score


def _bin2(im: np.ndarray) -> np.ndarray:
    """Average-pooling downscale by factor 2 for robust pyramid levels."""
    h, w = im.shape
    h2, w2 = h // 2, w // 2
    im_c = im[: 2 * h2, : 2 * w2]
    return im_c.reshape(h2, 2, w2, 2).mean(axis=(1, 3))


def _normalize01(img: np.ndarray) -> np.ndarray:
    m, M = np.min(img), np.max(img)
    if M > m:
        return (img - m) / (M - m)
    return np.zeros_like(img)


def align_phasecorr(
    moving: np.ndarray,
    reference: np.ndarray,
    metric: str = "ncc",
    border: float | int = 0.1,
    upsample: int = 1,
    refine_radius: int = 6,
) -> Tuple[int, int]:
    """Estimate shift via phase correlation.

    Returns integer (dy, dx).
    """
    # Preprocess for robustness if needed
    if metric in ("edges", "edges5"):
        m = _prepare_for_metric(moving, metric)
        r = _prepare_for_metric(reference, metric)
    else:
        m, r = moving, reference

    # Crop borders inside arrays to avoid frame bias
    m_c = crop_border(m, border)
    r_c = crop_border(r, border)

    # Phase correlation (subpixel allowed by upsample)
    shift, _, _ = phase_cross_correlation(r_c, m_c, upsample_factor=max(1, upsample))
    dy0, dx0 = int(round(shift[0])), int(round(shift[1]))

    # Local refinement on original (non-cropped) arrays around estimate
    dy, dx, _ = exhaustive_search(
        moving,
        reference,
        range(dy0 - refine_radius, dy0 + refine_radius + 1),
        range(dx0 - refine_radius, dx0 + refine_radius + 1),
        metric if metric != "auto" else "ncc",
        border,
    )
    return dy, dx


def align_pyramid(
    moving: np.ndarray,
    reference: np.ndarray,
    max_radius: int = 15,
    metric: str = "ncc",
    border: float | int = 0.1,
    scale: int = 2,
    min_size: int = 400,
) -> Tuple[int, int]:
    """Coarse-to-fine alignment using an image pyramid.

    Returns integer-pixel shift (dy, dx) to align moving to reference.
    """
    h, w = reference.shape

    # If already small enough, just exhaustive within given radius
    if min(h, w) <= min_size:
        radius = max(1, max_radius)
        dy, dx, _ = exhaustive_search(
            moving, reference, range(-radius, radius + 1), range(-radius, radius + 1), metric, border
        )
        return dy, dx

    # Build simple average-pooling pyramid (coarsest first)
    pyr_mov = [moving]
    pyr_ref = [reference]
    while min(pyr_mov[-1].shape) > min_size:
        pyr_mov.append(_bin2(pyr_mov[-1]))
        pyr_ref.append(_bin2(pyr_ref[-1]))
    # Reverse to start at coarsest
    pyr_mov = pyr_mov[::-1]
    pyr_ref = pyr_ref[::-1]

    # Start from zero shift at coarsest
    dy_acc, dx_acc = 0, 0
    levels = len(pyr_mov)

    for lvl, (m_lvl, r_lvl) in enumerate(zip(pyr_mov, pyr_ref)):
        # Dynamic per-level radius
        scale_rem = 2 ** (levels - 1 - lvl)
        level_radius = int(np.ceil(max_radius / max(1, scale_rem)))
        level_radius = int(np.clip(level_radius, 10, 30))
        if lvl == 0:
            dy, dx, _ = exhaustive_search(
                m_lvl, r_lvl, range(-level_radius, level_radius + 1), range(-level_radius, level_radius + 1), metric, border
            )
            dy_acc, dx_acc = dy, dx
        else:
            dy0, dx0 = dy_acc * 2, dx_acc * 2
            dy, dx, _ = exhaustive_search(
                m_lvl,
                r_lvl,
                range(dy0 - level_radius, dy0 + level_radius + 1),
                range(dx0 - level_radius, dx0 + level_radius + 1),
                metric,
                border,
            )
            dy_acc, dx_acc = dy, dx

    return int(dy_acc), int(dx_acc)


def _save_debug(
    out_dir: str,
    base: str,
    b: np.ndarray,
    g: np.ndarray,
    r: np.ndarray,
    g_aligned: np.ndarray,
    r_aligned: np.ndarray,
    metric: str,
    g_vec: Tuple[int, int],
    r_vec: Tuple[int, int],
    border: float | int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Raw channels
    imsave(os.path.join(out_dir, f"{base}_b.jpg"), img_as_ubyte(_normalize01(b)))
    imsave(os.path.join(out_dir, f"{base}_g.jpg"), img_as_ubyte(_normalize01(g)))
    imsave(os.path.join(out_dir, f"{base}_r.jpg"), img_as_ubyte(_normalize01(r)))

    # Aligned overlays: show differences to B
    diff_g = np.abs(g_aligned - b)
    diff_r = np.abs(r_aligned - b)
    imsave(os.path.join(out_dir, f"{base}_diff_gb.jpg"), img_as_ubyte(_normalize01(diff_g)))
    imsave(os.path.join(out_dir, f"{base}_diff_rb.jpg"), img_as_ubyte(_normalize01(diff_r)))

    # Edge maps used (if edges metric)
    if metric.startswith("edges"):
        g_edge = _prepare_for_metric(g, metric)
        r_edge = _prepare_for_metric(r, metric)
        b_edge = _prepare_for_metric(b, metric)
        imsave(os.path.join(out_dir, f"{base}_b_edges.jpg"), img_as_ubyte(_normalize01(b_edge)))
        imsave(os.path.join(out_dir, f"{base}_g_edges.jpg"), img_as_ubyte(_normalize01(g_edge)))
        imsave(os.path.join(out_dir, f"{base}_r_edges.jpg"), img_as_ubyte(_normalize01(r_edge)))

    # Small score map around optimum to visualize landscape
    def score_map(m: np.ndarray, ref: np.ndarray, dy0: int, dx0: int) -> np.ndarray:
        win = 8
        scores = np.zeros((2 * win + 1, 2 * win + 1), dtype=float)
        for i, dy in enumerate(range(dy0 - win, dy0 + win + 1)):
            for j, dx in enumerate(range(dx0 - win, dx0 + win + 1)):
                scores[i, j] = _eval_alignment(m, ref, dy, dx, border)
        return scores

    sm_g = _normalize01(score_map(g, b, g_vec[0], g_vec[1]))
    sm_r = _normalize01(score_map(r, b, r_vec[0], r_vec[1]))
    # Stretch to image and save
    imsave(os.path.join(out_dir, f"{base}_score_g.jpg"), img_as_ubyte(sm_g))
    imsave(os.path.join(out_dir, f"{base}_score_r.jpg"), img_as_ubyte(sm_r))


def _score_grid(
    moving: np.ndarray,
    reference: np.ndarray,
    radius: int,
    metric: str,
    border: float | int,
):
    dy_vals = list(range(-radius, radius + 1))
    dx_vals = list(range(-radius, radius + 1))
    scores = np.zeros((len(dy_vals), len(dx_vals)), dtype=float)
    for i, dy in enumerate(dy_vals):
        for j, dx in enumerate(dx_vals):
            ys_m, xs_m, ys_r, xs_r = _overlap_slices(reference.shape[0], reference.shape[1], dy, dx)
            a = crop_border(moving[ys_m, xs_m], border)
            b = crop_border(reference[ys_r, xs_r], border)
            scores[i, j] = _metric_score(a, b, metric if metric != "edges" else "ssd")
    return np.array(dy_vals), np.array(dx_vals), scores


def _save_debug_deep(
    out_dir: str,
    base: str,
    b_for_align: np.ndarray,
    g_for_align: np.ndarray,
    r_for_align: np.ndarray,
    used_metric: str,
    border: float | int,
    radius: int,
    topk: int = 5,
):
    os.makedirs(out_dir, exist_ok=True)

    # Full score grids
    for ch_name, m in ("g", g_for_align), ("r", r_for_align):
        dyv, dxv, grid = _score_grid(m, b_for_align, radius, used_metric, border)
        # Normalize and enlarge for visibility
        grid_n = _normalize01(grid)
        from skimage.transform import resize
        big = resize(grid_n, (grid_n.shape[0] * 8, grid_n.shape[1] * 8), anti_aliasing=False, preserve_range=True)
        imsave(os.path.join(out_dir, f"{base}_grid_{ch_name}.jpg"), img_as_ubyte(big))
        # Save CSV for exact numbers
        csv_path = os.path.join(out_dir, f"{base}_grid_{ch_name}.csv")
        with open(csv_path, "w") as f:
            f.write("dy,dx,score\n")
            # Flatten sorted by score desc
            flat = [
                (int(dy), int(dx), float(grid[i, j]))
                for i, dy in enumerate(dyv)
                for j, dx in enumerate(dxv)
            ]
            flat.sort(key=lambda t: t[2], reverse=True)
            for dy, dx, s in flat:
                f.write(f"{dy},{dx},{s}\n")

        # Also save quick previews for top-K candidates using diff maps
        hh, ww = b_for_align.shape
        cy, cx = hh // 2, ww // 2
        win = min(hh, ww, 200) // 2
        tiles = []
        for (dy, dx, _) in flat[:topk]:
            ys_m, xs_m, ys_r, xs_r = _overlap_slices(hh, ww, dy, dx)
            a = g_for_align if ch_name == "g" else r_for_align
            shifted = shift_image(a, dy, dx)
            diff = np.abs(shifted - b_for_align)
            crop = diff[cy - win : cy + win, cx - win : cx + win]
            tiles.append(img_as_ubyte(_normalize01(crop)))
        if tiles:
            strip = np.concatenate(tiles, axis=1)
            imsave(os.path.join(out_dir, f"{base}_top{topk}_diff_strip_{ch_name}.jpg"), strip)


def align_single_scale(
    moving: np.ndarray, reference: np.ndarray, radius: int = 15, metric: str = "ncc", border: float | int = 0.1
) -> Tuple[int, int]:
    dy, dx, _ = exhaustive_search(
        moving, reference, range(-radius, radius + 1), range(-radius, radius + 1), metric, border
    )
    return dy, dx


def shift_image(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift an image by (dy, dx) using wraparound (np.roll). For visualization; border cropping recommended."""
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


def split_bgr_stack(im: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a vertically stacked B, G, R image into separate channels."""
    h = im.shape[0] // 3
    b = im[:h, :]
    g = im[h : 2 * h, :]
    r = im[2 * h : 3 * h, :]
    return b, g, r


def _align_to_anchor(
    anchor: str,
    b: np.ndarray,
    g: np.ndarray,
    r: np.ndarray,
    use_pyramid: bool,
    metric: str,
    max_radius: int,
    border: float | int,
    pyr_min_size: int,
) -> tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Align two channels to the chosen anchor channel and return stacked RGB and shifts for G and R relative to B.

    Note: For reporting consistency, always report (dy,dx) for G and R that were actually APPLIED to align to B.
    """
    # helper
    def do_align(m: np.ndarray, ref: np.ndarray) -> Tuple[int, int]:
        if use_pyramid:
            return align_pyramid(m, ref, max_radius=max_radius, metric=metric, border=border, min_size=pyr_min_size)
        else:
            radius = min(30, max_radius)
            return align_single_scale(m, ref, radius=radius, metric=metric, border=border)

    if anchor == 'b':
        dy_g, dx_g = do_align(g, b)
        dy_r, dx_r = do_align(r, b)
        g_al = shift_image(g, -dy_g, -dx_g)
        r_al = shift_image(r, -dy_r, -dx_r)
        rgb = np.dstack([r_al, g_al, b])
        return rgb, (-dy_g, -dx_g), (-dy_r, -dx_r)
    elif anchor == 'g':
        dy_b, dx_b = do_align(b, g)
        dy_r, dx_r = do_align(r, g)
        b_al = shift_image(b, -dy_b, -dx_b)
        r_al = shift_image(r, -dy_r, -dx_r)
        g_vec = (dy_b, dx_b)  # applied to G RELATIVE TO B if had moved G
        rgb = np.dstack([r_al, g, b_al])
        return rgb, g_vec, (-dy_r, -dx_r)
    else:  # anchor == 'r'
        dy_b, dx_b = do_align(b, r)
        dy_g, dx_g = do_align(g, r)
        b_al = shift_image(b, -dy_b, -dx_b)
        g_al = shift_image(g, -dy_g, -dx_g)
        rgb = np.dstack([r, g_al, b_al])
        return rgb, (-dy_g + dy_b, -dx_g + dx_b), (dy_b, dx_b)


def _eval_alignment(moving: np.ndarray, reference: np.ndarray, dy: int, dx: int, border: float | int = 0.1) -> float:
    """Return negative SSD after applying (dy, dx); higher is better (less error)."""
    h, w = reference.shape
    ys_m, xs_m, ys_r, xs_r = _overlap_slices(h, w, dy, dx)
    a = moving[ys_m, xs_m]
    b = reference[ys_r, xs_r]
    a = crop_border(a, border)
    b = crop_border(b, border)
    diff = a - b
    return -float(np.sum(diff * diff))


def colorize(
    im_stack: np.ndarray,
    use_pyramid: bool = True,
    metric: str = "ncc",
    max_radius: int = 200,
    border: float | int = 0.1,
    pyramid_min_size: int = 400,
    pre_crop: float = 0.07,
    auto_crop_output: bool = True,
    anchor: str = 'b',
    method: str = 'pyramid',  # 'pyramid' | 'single' | 'phase' | 'phase-refine'
    detect_borders: bool = False,
    detect_limit_frac: float = 0.25,
    detect_inward: int = 1,
    debug_dir: str | None = None,
    ) -> tuple[np.ndarray, Tuple[int, int], Tuple[int, int], str]:
    """Align channels and return color image plus displacement vectors (APPLIED) for G and R relative to B.

    Returns: (rgb_image, g_applied_shift, r_applied_shift, used_metric)
    """
    b, g, r = split_bgr_stack(im_stack)

    # adaptive border detection per channel with intersection
    if detect_borders:
        bboxes = []
        borders_dbg = None
        if debug_dir is not None:
            borders_dbg = os.path.join(debug_dir, "borders")
            os.makedirs(borders_dbg, exist_ok=True)
        for ch, arr in (("b", b), ("g", g), ("r", r)):
            dbg = os.path.join(borders_dbg, ch) if borders_dbg else None
            t, bt, l, rgt = detect_black_bbox(
                arr, limit_frac=detect_limit_frac, inward=detect_inward, debug_dir=dbg, tag=f"{ch}"
            )
            bboxes.append((t, bt, l, rgt))
        h, w = b.shape
        top = max(bb[0] for bb in bboxes)
        bottom = min(bb[1] for bb in bboxes)
        left = max(bb[2] for bb in bboxes)
        right = min(bb[3] for bb in bboxes)
        if 0 <= top < bottom <= h and 0 <= left < right <= w:
            if borders_dbg is not None:
                with open(os.path.join(borders_dbg, "intersection_box.txt"), "w") as f:
                    f.write(f"top={top}, bottom={bottom}, left={left}, right={right}\n")
            b = b[top:bottom, left:right]
            g = g[top:bottom, left:right]
            r = r[top:bottom, left:right]

    # crop borders before alignment to ignore black frames
    if pre_crop and pre_crop > 0 and not detect_borders:
        def pc(x: np.ndarray) -> np.ndarray:
            return crop_border(x, pre_crop)
        b_for_align, g_for_align, r_for_align = pc(b), pc(g), pc(r)
    else:
        b_for_align, g_for_align, r_for_align = b, g, r

    # Allow auto metric selection by wrapping the metric argument
    metric_to_use = metric
    # ---------- Auto metric selection (robust for Emir) ----------
    def choose_metric(m: np.ndarray, ref: np.ndarray) -> str:
        if metric != "auto":
            return metric
        candidates = ["ncc", "edges5"]
        best_m = candidates[0]
        best_s = -np.inf
        for met in candidates:
            # align with 'met' at the chosen method
            if use_pyramid:
                dy, dx = align_pyramid(m, ref, max_radius=max_radius, metric=met,
                                       border=border, min_size=pyramid_min_size)
            else:
                dy, dx = align_single_scale(m, ref, radius=min(30, max_radius),
                                            metric=met, border=border)
            # score in the SAME feature space as the metric
            s = _score_at(m, ref, dy, dx, met, border)
            if s > best_s:
                best_s, best_m = s, met
        return best_m
    # -------------------------------------------------------------

    # Align channels to chosen anchor with the decided metric
    def align_func(m, ref):
        if method == 'phase' or method == 'phase-refine':
            refine_rad = 10 if method == 'phase-refine' else 4
            return align_phasecorr(m, ref, metric=metric_to_use, border=border, upsample=5, refine_radius=refine_rad)
        if method == 'single':
            return align_single_scale(m, ref, radius=min(30, max_radius), metric=metric_to_use, border=border)
        # default pyramid
        return align_pyramid(m, ref, max_radius=max_radius, metric=metric_to_use, border=border, min_size=pyramid_min_size)

    if anchor == 'b':
        g_dy, g_dx = align_func(g_for_align, b_for_align)
        r_dy, r_dx = align_func(r_for_align, b_for_align)
        # APPLY the negative (move channels toward B)
        g_shift = (-g_dy, -g_dx)
        r_shift = (-r_dy, -r_dx)
        g_aligned = shift_image(g, g_shift[0], g_shift[1])
        r_aligned = shift_image(r, r_shift[0], r_shift[1])
        b_shift = (0, 0)
    elif anchor == 'g':
        b_dy, b_dx = align_func(b_for_align, g_for_align)
        r_dy, r_dx = align_func(r_for_align, g_for_align)
        # Move B and R toward G
        b = shift_image(b, -b_dy, -b_dx)
        g_aligned = g
        r_aligned = shift_image(r, -r_dy, -r_dx)
        # Report applied shifts RELATIVE TO B:
        g_shift = ( b_dy,  b_dx)   # equivalent net motion of G relative to B
        r_shift = (-r_dy, -r_dx)
        b_shift = (-b_dy, -b_dx)
    else:  # anchor == 'r'
        b_dy, b_dx = align_func(b_for_align, r_for_align)
        g_dy2, g_dx2 = align_func(g_for_align, r_for_align)
        b = shift_image(b, -b_dy, -b_dx)
        g_aligned = shift_image(g, -g_dy2, -g_dx2)
        r_aligned = r
        # Applied (reported) shifts relative to B:
        b_shift = (-b_dy, -b_dx)
        g_shift = (-g_dy2, -g_dx2)
        r_shift = (0, 0)

    if auto_crop_output:
        # Compute overlap crop to remove wrap-around borders
        def pos(x): return x if x > 0 else 0
        def neg(x): return -x if x < 0 else 0
        top = max(pos(b_shift[0]), pos(g_shift[0]), pos(r_shift[0]))
        bottom = max(neg(b_shift[0]), neg(g_shift[0]), neg(r_shift[0]))
        left = max(pos(b_shift[1]), pos(g_shift[1]), pos(r_shift[1]))
        right = max(neg(b_shift[1]), neg(g_shift[1]), neg(r_shift[1]))
        def crop_common(x: np.ndarray) -> np.ndarray:
            h, w = x.shape[:2]
            return x[top:h-bottom if bottom > 0 else h, left:w-right if right > 0 else w]
        b_c = crop_common(b)
        g_c = crop_common(g_aligned)
        r_c = crop_common(r_aligned)
        rgb = np.dstack([r_c, g_c, b_c])
    else:
        rgb = np.dstack([r_aligned, g_aligned, b])

    # Return the shifts applied
    return rgb, g_shift, r_shift, metric_to_use


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prokudin-Gorskii Colorizer (CS180/CS280A)")
    p.add_argument(
        "--input",
        required=True,
        help="Path to input glass plate image (e.g., 'cs180 proj1 data/monastery.jpg')",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to ./output/<basename>_color.jpg",
    )
    p.add_argument(
        "--metric",
        choices=["ssd", "ncc", "edges", "edges5", "auto"],
        default="ncc",
        help="Scoring metric: ssd, ncc, edges (Sobel), edges5 (Gaussian+Sobel), or auto (choose best)",
    )
    p.add_argument(
        "--no-pyramid",
        action="store_true",
        help="Deprecated convenience flag. If set and --method not provided, switch to --method single.",
    )
    p.add_argument(
        "--max-radius",
        type=int,
        default=200,
        help="Maximum search radius at the finest level (pyramid) or single-scale",
    )
    p.add_argument(
        "--border",
        type=float,
        default=0.1,
        help="Border to ignore during scoring; <1 means fraction, >=1 means pixels",
    )
    p.add_argument(
        "--pyr-min-size",
        type=int,
        default=400,
        help="Minimum dimension at which to stop recursing in pyramid",
    )
    p.add_argument(
        "--pre-crop",
        type=float,
        default=0.08,
        help="Fractional border to crop from each channel before alignment (helps ignore black frames)",
    )
    ac = p.add_mutually_exclusive_group()
    ac.add_argument("--auto-crop", dest="auto_crop", action="store_true", help="Crop final image to common overlap")
    ac.add_argument("--no-auto-crop", dest="auto_crop", action="store_false", help="Disable overlap auto-cropping")
    p.set_defaults(auto_crop=True)
    p.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate debug images (channels, diffs, score maps) to output/debug/",
    )
    p.add_argument(
        "--anchor",
        choices=['b', 'g', 'r'],
        default='b',
        help="Which channel to align others to (default: b). Sometimes 'g' works better.",
    )
    p.add_argument(
        "--method",
        choices=['pyramid', 'single', 'phase', 'phase-refine'],
        default='pyramid',
        help="Alignment method: pyramid (default), single-scale, phase correlation, or phase with wider refinement",
    )
    bd = p.add_mutually_exclusive_group()
    bd.add_argument("--detect-borders", dest="detect_borders", action="store_true", help="Auto-detect per-channel borders using edges")
    bd.add_argument("--no-detect-borders", dest="detect_borders", action="store_false", help="Disable border detection (default)")
    p.set_defaults(detect_borders=False)
    p.add_argument("--detect-limit-frac", type=float, default=0.25, help="Frac of image near edges to search for border lines")
    p.add_argument("--detect-inward", type=int, default=1, help="Move crop inward by this many pixels after detecting lines")
    p.add_argument(
        "--debug-deep",
        action="store_true",
        help="For single-scale: dump full score grids, CSVs, and candidate previews",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Convenience: if someone uses --no-pyramid but doesn't set --method, switch to single-scale.
    if args.no_pyramid and args.method == "pyramid":
        args.method = "single"

    # Read and normalize image
    im = img_as_float(imread(args.input))
    # Ensure grayscale 2D
    if im.ndim == 3:
        # Some readers may load grayscale as (H, W, 3) duplicates; convert to single channel
        if im.shape[2] == 3 and np.allclose(im[..., 0], im[..., 1]) and np.allclose(im[..., 1], im[..., 2]):
            im = im[..., 0]
        else:
            raise ValueError("Input must be a vertically stacked grayscale image with B,G,R channels.")

    use_pyr = (args.method == "pyramid")
    base = os.path.splitext(os.path.basename(args.input))[0]
    dbg_dir = os.path.join("output", "debug", base) if args.debug else None
    rgb, g_vec, r_vec, used_metric = colorize(
        im,
        use_pyramid=use_pyr,
        metric=args.metric,
        max_radius=args.max_radius,
        border=args.border,
        pyramid_min_size=args.pyr_min_size,
        pre_crop=args.pre_crop,
        auto_crop_output=args.auto_crop,
        anchor=args.anchor,
        method=args.method,
        detect_borders=args.detect_borders,
        detect_limit_frac=args.detect_limit_frac,
        detect_inward=args.detect_inward,
        debug_dir=dbg_dir,
    )

    # Prepare output path
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_dir = os.path.join(".", "output")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}_color.jpg")
    else:
        out_path = args.output
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Convert to 8-bit for Pillow/JPEG compatibility
    imsave(out_path, img_as_ubyte(np.clip(rgb, 0, 1)))

    if args.debug:
        # Use the applied shifts for visibility in debug
        b, g, r = split_bgr_stack(im)
        g_shifted = shift_image(g, g_vec[0], g_vec[1])
        r_shifted = shift_image(r, r_vec[0], r_vec[1])
        _save_debug(dbg_dir, base, b, g, r, g_shifted, r_shifted, used_metric, g_vec, r_vec, args.border)

        # Deep debug for single-scale mode: dump full score grids and top-K previews
        if args.debug_deep and args.method in ("single",):
            def pc(x):
                return crop_border(x, args.pre_crop) if args.pre_crop and args.pre_crop > 0 else x
            b_a, g_a, r_a = pc(b), pc(g), pc(r)
            radius = min(30, args.max_radius)
            deep_dir = os.path.join(dbg_dir, "deep")
            _save_debug_deep(deep_dir, base, b_a, g_a, r_a, used_metric, args.border, radius, topk=6)

    # Report settings and displacements
    print(f"Method: {args.method}, Anchor: {args.anchor}, Metric: {used_metric}, Pre-crop: {args.pre_crop}, Border: {args.border}")
    print(f"Applied shift to G (relative to B): dy={g_vec[0]}, dx={g_vec[1]}")
    print(f"Applied shift to R (relative to B): dy={r_vec[0]}, dx={r_vec[1]}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
