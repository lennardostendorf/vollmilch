"""Nucleus / cell segmentation with Cellpose-SAM in overlapping windows.

Each window is segmented independently; an object is kept by the window whose core
(the window minus half the overlap on interior sides) contains its centroid, so every
object is taken from exactly one window and is never cut at a window border as long as
the overlap is larger than the object diameter.
"""
from __future__ import annotations

import logging
import time

import numpy as np
from scipy import ndimage as ndi

log = logging.getLogger("vollmilch")


# ---------------------------------------------------------------------------- helpers
def percentile_range(arr: np.ndarray, mask: np.ndarray | None, pcts=(1, 99.8), max_samples=2_000_000, seed=0):
    vals = arr[mask] if mask is not None else arr.ravel()
    if vals.size == 0:
        return 0.0, 1.0
    if vals.size > max_samples:
        vals = np.random.default_rng(seed).choice(vals, max_samples, replace=False)
    lo, hi = np.percentile(vals.astype(np.float64), pcts)
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def window_starts(n: int, tile: int, overlap: int):
    if n <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, n - tile, step)) + [n - tile]
    return sorted(set(starts))


def window_cores(starts, tile, n):
    """Partition [0, n) into cores, one per window, split at the middle of each overlap."""
    ends = [min(s + tile, n) for s in starts]
    bounds = [0]
    for i in range(len(starts) - 1):
        bounds.append((starts[i + 1] + ends[i]) // 2)
    bounds.append(n)
    return [(bounds[i], bounds[i + 1]) for i in range(len(starts))]


def relabel_by_map(lab: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    return mapping[lab]


def object_centroids(lab: np.ndarray, n: int | None = None):
    n = int(lab.max()) if n is None else n
    yy, xx = np.nonzero(lab)
    ll = lab[yy, xx]
    cnt = np.bincount(ll, minlength=n + 1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        cy = np.bincount(ll, yy, minlength=n + 1) / cnt
        cx = np.bincount(ll, xx, minlength=n + 1) / cnt
    return cnt, cy, cx


# ---------------------------------------------------------------------------- backends
class CellposeBackend:
    def __init__(self, seg_cfg: dict, pixel_size_um: float):
        from cellpose import models

        gpu = seg_cfg.get("gpu", "auto")
        if gpu == "auto":
            try:
                import torch

                gpu = bool(torch.cuda.is_available() or (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()))
            except Exception:
                gpu = False
        kw = {"gpu": bool(gpu)}
        if seg_cfg.get("model"):
            kw["pretrained_model"] = seg_cfg["model"]
        t = time.time()
        self.model = models.CellposeModel(**kw)
        log.info("Cellpose model loaded (gpu=%s) in %.0f s", gpu, time.time() - t)
        d = seg_cfg.get("diameter_um")
        self.diameter = d / pixel_size_um if d else None
        self.flow_threshold = seg_cfg.get("flow_threshold", 0.4)
        self.cellprob_threshold = seg_cfg.get("cellprob_threshold", 0.0)
        self.batch_size = seg_cfg.get("batch_size", 8)

    def __call__(self, x: np.ndarray, min_size: int) -> np.ndarray:
        """x: H x W x 3 float32, already normalised to ~[0, 1]."""
        out = self.model.eval(
            x, channel_axis=2, normalize=False, diameter=self.diameter,
            flow_threshold=self.flow_threshold, cellprob_threshold=self.cellprob_threshold,
            min_size=int(min_size), batch_size=self.batch_size,
        )
        return np.asarray(out[0]).astype(np.int32)


class WatershedBackend:
    """Classical nuclear segmentation (Otsu + distance-transform watershed). For quick
    tests or machines without a usable Cellpose installation; nuclei only."""

    def __init__(self, seg_cfg: dict, pixel_size_um: float):
        d_um = seg_cfg.get("diameter_um") or 7.0
        self.r_px = max(2, int(round(d_um / pixel_size_um / 2)))
        self.threshold = None

    def set_threshold(self, x_norm_sample: np.ndarray):
        from skimage.filters import threshold_otsu

        self.threshold = float(threshold_otsu(x_norm_sample))

    def __call__(self, x: np.ndarray, min_size: int) -> np.ndarray:
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed

        img = ndi.gaussian_filter(x[..., 0].astype(np.float32), 1.0)
        thr = self.threshold if self.threshold is not None else img.mean() + img.std()
        fg = img > thr
        fg = ndi.binary_opening(fg, iterations=1)
        if not fg.any():
            return np.zeros(fg.shape, np.int32)
        dist = ndi.distance_transform_edt(fg)
        dist = ndi.gaussian_filter(dist, 1.0)
        pk = peak_local_max(dist, min_distance=max(2, int(self.r_px * 0.7)), labels=fg.astype(np.int32), exclude_border=False)
        markers = np.zeros(fg.shape, np.int32)
        markers[tuple(pk.T)] = np.arange(1, len(pk) + 1)
        lab = watershed(-dist, markers, mask=fg)
        cnt = np.bincount(lab.ravel())
        small = np.flatnonzero(cnt < min_size)
        if small.size:
            lab[np.isin(lab, small)] = 0
        return lab.astype(np.int32)


def make_backend(seg_cfg, pixel_size_um):
    b = seg_cfg.get("backend", "cellpose")
    if b == "cellpose":
        try:
            return CellposeBackend(seg_cfg, pixel_size_um)
        except ImportError as e:
            raise ImportError("Cellpose is not installed: pip install 'vollmilch[segment]' (or backend: watershed)") from e
    if b == "watershed":
        return WatershedBackend(seg_cfg, pixel_size_um)
    raise ValueError(f"unknown segmentation.backend {b!r}")


# ---------------------------------------------------------------------------- tiling
def segment_tiled(get_input, shape, mask, predict, tile=2048, overlap=160, min_size=15, label=""):
    """get_input(y0, y1, x0, x1) -> h x w x 3 float32. Returns uint32 labels of `shape`."""
    H, W = shape
    ys, xs = window_starts(H, tile, overlap), window_starts(W, tile, overlap)
    ycores, xcores = window_cores(ys, tile, H), window_cores(xs, tile, W)
    out = np.zeros((H, W), np.uint32)
    next_id = 1
    n_win = len(ys) * len(xs)
    t0 = time.time()
    k = 0
    for iy, y0 in enumerate(ys):
        for ix, x0 in enumerate(xs):
            k += 1
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            if mask is not None and not mask[y0:y1, x0:x1].any():
                continue
            x = get_input(y0, y1, x0, x1)
            if mask is not None:
                x = x * mask[y0:y1, x0:x1, None]
            if float(x.max()) <= 0:
                continue
            lab = predict(x, min_size)
            n = int(lab.max())
            if n == 0:
                continue
            cnt, cy, cx = object_centroids(lab, n)
            cy, cx = cy + y0, cx + x0
            (cy0, cy1), (cx0, cx1) = ycores[iy], xcores[ix]
            keep = (cnt >= min_size) & (cy >= cy0) & (cy < cy1) & (cx >= cx0) & (cx < cx1)
            keep[0] = False
            ids = np.flatnonzero(keep)
            mapping = np.zeros(n + 1, np.uint32)
            mapping[ids] = np.arange(next_id, next_id + len(ids), dtype=np.uint32)
            next_id += len(ids)
            new = mapping[lab]
            g = out[y0:y1, x0:x1]
            w = (new > 0) & (g == 0)
            g[w] = new[w]
            if n_win > 1:
                log.info("  %s window %d/%d: %d objects (%.0f s)", label, k, n_win, len(ids), time.time() - t0)
    return out


def expand_labels_tiled(lab: np.ndarray, distance: float, tile: int = 4096) -> np.ndarray:
    """skimage expand_labels in windows (expansion is local), to bound memory on large regions."""
    from skimage.segmentation import expand_labels

    if distance <= 0:
        return lab.copy()
    H, W = lab.shape
    if H * W <= tile * tile:
        return expand_labels(lab, distance).astype(lab.dtype)
    m = int(np.ceil(distance)) + 2
    out = np.zeros_like(lab)
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            a0, b0, a1, b1 = max(0, y0 - m), max(0, x0 - m), min(H, y1 + m), min(W, x1 + m)
            e = expand_labels(lab[a0:a1, b0:b1], distance)
            out[y0:y1, x0:x1] = e[y0 - a0:y1 - a0, x0 - b0:x1 - b0]
    return out


def sequential(lab: np.ndarray) -> np.ndarray:
    from skimage.segmentation import relabel_sequential

    return relabel_sequential(lab)[0].astype(np.uint32)


def match_nuclei_cells(nl, cl, mask, expand_px, keep_unmatched=True, require_nucleus=False):
    """Pair whole-cell and nuclear masks. Returns (nuclei, cells) with shared ids;
    nuclei are clipped to their cell."""
    nmax, cmax = int(nl.max()), int(cl.max())
    n_area = np.bincount(nl.ravel(), minlength=nmax + 1)
    both = (nl > 0) & (cl > 0)
    key = nl[both].astype(np.int64) * (cmax + 1) + cl[both]
    uk, cnt = np.unique(key, return_counts=True)
    pn, pc = uk // (cmax + 1), uk % (cmax + 1)
    cell_nuc = np.zeros(cmax + 1, np.int64)
    if len(uk):
        o = np.lexsort((-cnt, pn))
        f = np.r_[True, pn[o][1:] != pn[o][:-1]]
        bn, bc, bcnt = pn[o][f], pc[o][f], cnt[o][f]
        ok = bcnt >= 0.5 * n_area[bn]
        bn, bc, bcnt = bn[ok], bc[ok], bcnt[ok]
        o2 = np.lexsort((-bcnt, bc))
        f2 = np.r_[True, bc[o2][1:] != bc[o2][:-1]] if len(bc) else np.zeros(0, bool)
        cell_nuc[bc[o2][f2]] = bn[o2][f2]
    c_area = np.bincount(cl.ravel(), minlength=cmax + 1)
    keep_c = c_area > 0
    keep_c[0] = False
    if require_nucleus:
        keep_c &= cell_nuc > 0
    cmap = np.zeros(cmax + 1, np.uint32)
    cmap[keep_c] = np.arange(1, keep_c.sum() + 1, dtype=np.uint32)
    cells = cmap[cl]
    own = (nl > 0) & (cl > 0) & (cell_nuc[cl] == nl)
    nuclei = np.where(own, cells, 0).astype(np.uint32)
    n_next = int(keep_c.sum()) + 1
    n_unmatched = 0
    if keep_unmatched:
        matched = np.zeros(nmax + 1, bool)
        matched[cell_nuc[cell_nuc > 0]] = True
        bg_overlap = np.bincount(nl[(nl > 0) & (cl == 0)], minlength=nmax + 1)
        cand = (~matched) & (bg_overlap >= 0.5 * n_area) & (n_area > 0)
        cand[0] = False
        ids = np.flatnonzero(cand)
        if len(ids):
            umap_ = np.zeros(nmax + 1, np.uint32)
            umap_[ids] = np.arange(n_next, n_next + len(ids), dtype=np.uint32)
            un = np.where(cl == 0, umap_[nl], 0).astype(np.uint32)
            nuclei = np.where(un > 0, un, nuclei)
            exp = expand_labels_tiled(un, expand_px)
            free = (cells == 0) & (exp > 0)
            if mask is not None:
                free &= mask
            cells = np.where(free, exp, cells).astype(np.uint32)
            n_unmatched = len(ids)
    log.info("  matched %d cells with nuclei, %d cells without nucleus, %d unmatched nuclei kept as expanded cells",
             int((cell_nuc[keep_c] > 0).sum()), int((cell_nuc[keep_c] == 0).sum()), n_unmatched)
    return nuclei, cells


def finalize(nuclei, cells, mask, min_nuc_px, min_cell_px):
    """Drop objects with centroid outside the region or too small; make ids 1..N shared."""
    from .features import label_moments

    n = int(max(nuclei.max(), cells.max()))
    c_cnt, cy, cx, *_ = label_moments(cells, n)
    n_cnt = label_moments(nuclei, n)[0]
    keep = c_cnt >= min_cell_px
    keep &= (n_cnt == 0) | (n_cnt >= min_nuc_px)
    if mask is not None:
        iy = np.clip(np.nan_to_num(np.round(cy), nan=0).astype(int), 0, mask.shape[0] - 1)
        ix = np.clip(np.nan_to_num(np.round(cx), nan=0).astype(int), 0, mask.shape[1] - 1)
        keep &= mask[iy, ix]
    keep[0] = False
    m = np.zeros(n + 1, np.uint32)
    m[keep] = np.arange(1, keep.sum() + 1, dtype=np.uint32)
    return m[nuclei], m[cells]


def segment_region(source, region, cfg, backend=None):
    """Returns (nuclei, cells) uint32 label images of the region bounding box (shared ids)."""
    seg = cfg["segmentation"]
    px = source.pixel_size_um
    mask = region.mask()
    bbox = region.bbox
    pcts = tuple(seg.get("normalize_percentiles", (1, 99.8)))
    backend = backend or make_backend(seg, px)

    def load_norm(ch):
        a = source.read(region.image, ch, bbox)
        lo, hi = percentile_range(a, mask, pcts)
        return a, lo, hi

    dapi, dlo, dhi = load_norm(seg["nuclear_channel"])
    min_nuc = max(1, int(round(seg.get("min_nucleus_area_um2", 8) / px**2)))
    min_cell = max(1, int(round(seg.get("min_cell_area_um2", 15) / px**2)))

    def nuc_input(y0, y1, x0, x1):
        x = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
        x[..., 0] = (dapi[y0:y1, x0:x1].astype(np.float32) - dlo) / (dhi - dlo)
        return np.clip(x, 0, None, out=x)

    if isinstance(backend, WatershedBackend):
        samp = dapi[mask] if mask.any() else dapi.ravel()
        samp = samp[np.random.default_rng(0).choice(samp.size, min(samp.size, 1_000_000), replace=False)]
        backend.set_threshold((samp.astype(np.float32) - dlo) / (dhi - dlo))

    t = time.time()
    nl = segment_tiled(nuc_input, mask.shape, mask, backend, seg["tile_px"], seg["overlap_px"], min_nuc, "nuclei")
    log.info("  %s: %d nuclei (%.0f s)", region.sample, int(nl.max()), time.time() - t)
    mode = seg["mode"]
    if mode == "nuclei":
        cl = nl.copy()
        min_cell = min_nuc
    elif mode == "expand":
        cl = expand_labels_tiled(nl, seg["expand_um"] / px)
        cl[~mask] = 0
    else:
        cyto = []
        for ch in seg["cyto_channels"]:
            cyto.append(load_norm(ch))
        comb = seg.get("cyto_combine", "max")

        def cell_input(y0, y1, x0, x1):
            x = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
            parts = [(a[y0:y1, x0:x1].astype(np.float32) - lo) / (hi - lo) for a, lo, hi in cyto]
            x[..., 0] = np.max(parts, axis=0) if comb == "max" else np.mean(parts, axis=0)
            x[..., 1] = (dapi[y0:y1, x0:x1].astype(np.float32) - dlo) / (dhi - dlo)
            return np.clip(x, 0, None, out=x)

        t = time.time()
        cl_raw = segment_tiled(cell_input, mask.shape, mask, backend, seg["tile_px"], seg["overlap_px"], min_cell, "cells")
        log.info("  %s: %d cells (%.0f s)", region.sample, int(cl_raw.max()), time.time() - t)
        nl, cl = match_nuclei_cells(nl, cl_raw, mask, seg["expand_um"] / px,
                                    seg.get("keep_unmatched_nuclei", True), seg.get("require_nucleus", False))
    nl, cl = finalize(nl, cl, mask, min_nuc, min_cell)
    return nl, cl
