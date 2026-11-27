"""Per-object features: morphology, nuclear texture, per-compartment marker statistics,
and multichannel thumbnails.

Memory: label images stay in memory; channels are read one at a time and all reductions
stream over blocks of rows, so peak memory is roughly labels + one channel + one block.
Only the `max` and `p90` statistics need a sorted pixel index per compartment (more memory)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

log = logging.getLogger("vollmilch")

BLOCK = 1024


def _blocks(H, block=BLOCK):
    for y in range(0, H, block):
        yield y, min(H, y + block)


def _bc(lab_flat, n, w=None):
    return np.bincount(lab_flat, weights=w, minlength=n + 1)[: n + 1]


def label_moments(lab: np.ndarray, n: int):
    """count, centroid (y, x) and central second moments per label, streamed over row blocks."""
    acc = np.zeros((6, n + 1))
    for y0, y1 in _blocks(lab.shape[0]):
        sub = lab[y0:y1]
        yy, xx = np.nonzero(sub)
        if not len(yy):
            continue
        ll = sub[yy, xx]
        yy = yy.astype(np.float64) + y0
        xx = xx.astype(np.float64)
        acc[0] += _bc(ll, n)
        acc[1] += _bc(ll, n, yy)
        acc[2] += _bc(ll, n, xx)
        acc[3] += _bc(ll, n, yy * yy)
        acc[4] += _bc(ll, n, xx * xx)
        acc[5] += _bc(ll, n, xx * yy)
    cnt = acc[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        my, mx = acc[1] / cnt, acc[2] / cnt
        cyy = np.clip(acc[3] / cnt - my * my, 0, None)
        cxx = np.clip(acc[4] / cnt - mx * mx, 0, None)
        cxy = acc[5] / cnt - mx * my
    return cnt, my, mx, cyy, cxx, cxy


def boundary_edges(lab: np.ndarray, n: int) -> np.ndarray:
    """Number of pixel edges between each label and anything else (image border included)."""
    H, W = lab.shape
    e = np.zeros(n + 1)

    def add(a, b):
        d = a != b
        e[:] += _bc(a[d], n) + _bc(b[d], n)

    zrow = np.zeros((1, W), lab.dtype)
    for y0, y1 in _blocks(H):
        sub = lab[y0:y1]
        z = np.zeros((sub.shape[0], 1), lab.dtype)
        add(np.hstack([z, sub]), np.hstack([sub, z]))
        nxt = lab[y0 + 1:y1 + 1]
        if y1 == H:
            nxt = np.vstack([nxt, zrow])
        add(sub, nxt)
        if y0 == 0:
            add(zrow, lab[:1])
    e[0] = 0
    return e


def _morph(lab: np.ndarray, n: int, px: float, prefix: str) -> pd.DataFrame:
    """Vectorised shape features (fast for millions of objects).
    Axes and eccentricity from second moments (as skimage), perimeter from the number of
    boundary pixel edges x pi/4, solidity from the convex hull of each object (OpenCV)."""
    import cv2

    cnt, _, _, cyy, cxx, cxy = label_moments(lab, n)
    tr, det = cxx + cyy, cxx * cyy - cxy**2
    disc = np.sqrt(np.clip(tr**2 / 4 - det, 0, None))
    l1, l2 = tr / 2 + disc, np.clip(tr / 2 - disc, 0, None)
    perim = boundary_edges(lab, n) * np.pi / 4
    hull = np.full(n + 1, np.nan)
    for oid, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None or oid > n:
            continue
        pts = cv2.findNonZero((lab[sl] == oid).astype(np.uint8))
        if pts is None:
            continue
        h = cv2.convexHull(pts)
        hull[oid] = cv2.contourArea(h) + cv2.arcLength(h, True) / 2 + 1
    a = cnt[1:]
    ok = a > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        cols = {
            "area_um2": a * px**2,
            "perimeter_um": perim[1:] * px,
            "circularity": np.clip(4 * np.pi * a / perim[1:] ** 2, 0, 1),
            "eccentricity": np.sqrt(np.clip(1 - l2[1:] / l1[1:], 0, 1)),
            "solidity": np.clip(a / hull[1:], 0, 1),
            "major_axis_um": 4 * np.sqrt(l1[1:]) * px,
            "minor_axis_um": 4 * np.sqrt(l2[1:]) * px,
            "aspect_ratio": np.sqrt(l1[1:] / np.where(l2[1:] > 0, l2[1:], np.nan)),
        }
    df = pd.DataFrame({f"{prefix}:{k}": np.where(ok, v, np.nan) for k, v in cols.items()})
    df.index = np.arange(1, n + 1)
    return df


class SortedIndex:
    """Pixel index sorted by object; only built when max/p90 are requested."""

    def __init__(self, lab: np.ndarray, n: int):
        flat = lab.ravel()
        pos = np.flatnonzero(flat)
        if flat.size < 2**31:
            pos = pos.astype(np.int32)
        ids = flat[pos]
        order = np.argsort(ids, kind="stable")
        self.pos = pos[order]
        self.ids = ids[order]
        del pos, ids, order
        self.n = n
        self.count = _bc(self.ids, n)
        self.starts = np.r_[0, np.cumsum(self.count[1:])][:-1]
        self.present = self.count[1:] > 0

    def max(self, flat_img):
        v = flat_img[self.pos]
        out = np.full(self.n, np.nan)
        st = self.starts[self.present]
        if st.size:
            out[self.present] = np.maximum.reduceat(v, st)
        return out

    def quantile(self, flat_img, q):
        v = flat_img[self.pos]
        o = np.lexsort((v, self.ids))
        vs = v[o]
        del o, v
        out = np.full(self.n, np.nan)
        idx = self.starts + np.floor(q * (self.count[1:] - 1)).astype(np.int64)
        out[self.present] = vs[idx[self.present]]
        return out


def _thumb_crops(img, centers, half):
    """img 2-D; centers (n, 2) int (y, x); returns (n, 2h, 2h) zero-padded crops."""
    H, W = img.shape
    s = 2 * half
    out = np.zeros((len(centers), s, s), img.dtype)
    for i, (cy, cx) in enumerate(centers):
        y0, x0 = cy - half, cx - half
        a0, b0 = max(0, y0), max(0, x0)
        a1, b1 = min(H, y0 + s), min(W, x0 + s)
        if a1 > a0 and b1 > b0:
            out[i, a0 - y0:a1 - y0, b0 - x0:b1 - x0] = img[a0:a1, b0:b1]
    return out


def _thumb_outlines(nuc, cell, ids, centers, half):
    """uint8 (n, s, s): bit 1 = nucleus outline, bit 2 = cell outline of the object itself."""
    nc = _thumb_crops(nuc, centers, half)
    cc = _thumb_crops(cell, centers, half)
    out = np.zeros(nc.shape, np.uint8)
    for i, oid in enumerate(ids):
        for crop, bit in ((nc[i], 1), (cc[i], 2)):
            m = crop == oid
            if m.any():
                b = m & ~ndi.binary_erosion(m, border_value=0)
                out[i][b] |= bit
    return out


def extract_features(source, region, nuc, cell, cfg, rng=None):
    """Returns (table, thumbs dict)."""
    px = source.pixel_size_um
    y0, y1, x0, x1 = region.bbox
    H = nuc.shape[0]
    n = int(cell.max())
    fc = cfg["features"]
    mode = cfg["segmentation"]["mode"]
    rng = rng or np.random.default_rng(0)

    chans = source.channel_names if fc["channels"] in ("all", None) else list(fc["channels"])
    excl = set(fc.get("exclude") or [])
    chans = [c for c in chans if c not in excl]
    for c in chans:
        source.channel_index(c)
    comps = list(dict.fromkeys(fc["compartments"]))
    if mode == "nuclei":
        comps = ["nucleus"] if any(c in comps for c in ("nucleus", "cell")) else []
    stats = list(fc["stats"])

    def comp_block(name, a, b):
        if name == "nucleus":
            return nuc[a:b]
        if name == "cell":
            return cell[a:b]
        return np.where(nuc[a:b] > 0, 0, cell[a:b])

    # --- position and morphology
    cnt_c, cy, cx, *_ = label_moments(cell, n)
    cy, cx = cy[1:], cx[1:]
    ids = np.arange(1, n + 1)
    df = pd.DataFrame({"object_id": ids, "x_px": cx + x0 + 0.5, "y_px": cy + y0 + 0.5})
    df["x_um"] = df["x_px"] * px
    df["y_um"] = df["y_px"] * px
    nm = _morph(nuc, n, px, "nucleus")
    df["has_nucleus"] = nm["nucleus:area_um2"].notna().values
    df = pd.concat([df, nm.reset_index(drop=True)], axis=1)
    if mode != "nuclei":
        cm = _morph(cell, n, px, "cell").drop(columns=["cell:major_axis_um", "cell:minor_axis_um"])
        df = pd.concat([df, cm.reset_index(drop=True)], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            df["cell:nc_ratio"] = df["nucleus:area_um2"] / df["cell:area_um2"]

    # pixel counts per compartment
    counts = {c: np.zeros(n + 1) for c in comps}
    for a, b in _blocks(H):
        for c in comps:
            counts[c] += _bc(comp_block(c, a, b).ravel(), n)
    sorted_idx = {}
    if any(s in ("max", "p90") for s in stats):
        for c in comps:
            lab = comp_block(c, 0, H) if c == "cytoplasm" else (nuc if c == "nucleus" else cell)
            sorted_idx[c] = SortedIndex(lab, n)

    # --- thumbnails setup
    tc = cfg["thumbnails"]
    t_chans = list(tc.get("channels") or [])
    for c in t_chans:
        source.channel_index(c)
    half = max(4, int(round(tc.get("size_um", 20) / px / 2)))
    n_t = min(n, int(tc.get("max_per_sample", 3000)))
    t_ids = np.sort(rng.choice(ids, n_t, replace=False)) if n_t < n else ids.copy()
    t_centers = (np.stack([np.round(cy[t_ids - 1]).astype(int), np.round(cx[t_ids - 1]).astype(int)], 1)
                 if n_t else np.zeros((0, 2), int))
    thumbs = {"ids": t_ids, "channels": t_chans, "half": half, "imgs": {}}
    if n_t:
        thumbs["outlines"] = _thumb_outlines(nuc, cell, t_ids, t_centers, half)

    # --- channel loop
    nuc_ch = cfg["segmentation"]["nuclear_channel"]
    texture = fc.get("nuclear_texture", True)
    read_list = list(dict.fromkeys(chans + t_chans + ([nuc_ch] if texture else [])))
    newcols = {}
    for ch in read_list:
        img = source.read(region.image, ch, (y0, y1, x0, x1))
        if ch in t_chans and n_t:
            thumbs["imgs"][ch] = _thumb_crops(img, t_centers, half)
        need_sd = "sd" in stats or (ch == nuc_ch and texture)
        sums = {c: np.zeros(n + 1) for c in comps}
        sq = {c: np.zeros(n + 1) for c in comps} if need_sd else {}
        tex = np.zeros((2, n + 1)) if (ch == nuc_ch and texture) else None
        if tex is not None:
            nsum, nsq = np.zeros(n + 1), np.zeros(n + 1)
        for a, b in _blocks(H):
            v = img[a:b].astype(np.float64).ravel()
            for c in comps:
                lb = comp_block(c, a, b).ravel()
                sums[c] += _bc(lb, n, v)
                if need_sd and c in sq:
                    sq[c] += _bc(lb, n, v * v)
            if tex is not None:
                lb = nuc[a:b].ravel()
                if "nucleus" not in comps:
                    nsum += _bc(lb, n, v)
                    nsq += _bc(lb, n, v * v)
                ha, hb = max(0, a - 1), min(H, b + 1)
                f = img[ha:hb].astype(np.float32)
                g = np.hypot(ndi.sobel(f, 0), ndi.sobel(f, 1))[a - ha:a - ha + (b - a)]
                mu = ndi.uniform_filter(f, 3)
                lsd = np.sqrt(np.clip(ndi.uniform_filter(f * f, 3) - mu * mu, 0, None))[a - ha:a - ha + (b - a)]
                tex[0] += _bc(lb, n, g.ravel().astype(np.float64))
                tex[1] += _bc(lb, n, lsd.ravel().astype(np.float64))
        with np.errstate(invalid="ignore", divide="ignore"):
            if ch in chans:
                for c in comps:
                    m = sums[c] / counts[c]
                    for st in stats:
                        if st == "mean":
                            r = m[1:]
                        elif st == "sd":
                            r = np.sqrt(np.clip(sq[c] / counts[c] - m * m, 0, None))[1:]
                        elif st == "max":
                            r = sorted_idx[c].max(img.ravel())
                        else:
                            r = sorted_idx[c].quantile(img.ravel(), 0.9)
                        newcols[f"{ch}:{c}:{st}"] = r
            if tex is not None:
                ncnt = counts["nucleus"] if "nucleus" in comps else _bc(nuc.ravel(), n)
                s1, s2 = (sums["nucleus"], sq["nucleus"]) if "nucleus" in comps else (nsum, nsq)
                m = s1 / ncnt
                sd = np.sqrt(np.clip(s2 / ncnt - m * m, 0, None))
                bright = np.zeros(n + 1)
                thr = 1.5 * m
                for a, b in _blocks(H):
                    lb = nuc[a:b].ravel()
                    v = img[a:b].ravel()
                    bright += _bc(lb, n, (v > thr[lb]).astype(np.float64))
                newcols["nucleus:dapi_integrated"] = s1[1:]
                newcols["nucleus:dapi_cv"] = (sd / m)[1:]
                newcols["nucleus:dapi_gradient"] = (tex[0] / ncnt / m)[1:]
                newcols["nucleus:dapi_local_sd"] = (tex[1] / ncnt / m)[1:]
                newcols["nucleus:dapi_bright_fraction"] = (bright / ncnt)[1:]
        del img
    df = pd.concat([df, pd.DataFrame(newcols)], axis=1)
    return df, thumbs


def qc_overlay(source, region, nuc, cell, cfg, out_path, size=600):
    """PNG of the nuclear channel with nucleus (cyan) and cell (magenta) outlines, around a
    dense part of the region."""
    from PIL import Image
    from skimage.segmentation import find_boundaries

    H, W = cell.shape
    if cell.max() == 0:
        return
    # densest window on a coarse grid
    f = max(1, min(H, W) // 64)
    small = (cell[::f, ::f] > 0).astype(np.float32)
    k = max(1, size // f)
    dens = ndi.uniform_filter(small, k)
    iy, ix = np.unravel_index(np.argmax(dens), dens.shape)
    cy, cx = iy * f, ix * f
    y0 = int(np.clip(cy - size // 2, 0, max(0, H - size)))
    x0 = int(np.clip(cx - size // 2, 0, max(0, W - size)))
    y1, x1 = min(H, y0 + size), min(W, x0 + size)
    by0, _, bx0, _ = region.bbox
    d = source.read(region.image, cfg["segmentation"]["nuclear_channel"], (by0 + y0, by0 + y1, bx0 + x0, bx0 + x1)).astype(np.float32)
    lo, hi = np.percentile(d, [1, 99.7])
    g = np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1)
    g = (np.sqrt(g) * 220).astype(np.uint8)
    rgb = np.stack([g, g, g], -1)
    nb = find_boundaries(nuc[y0:y1, x0:x1], mode="inner")
    cb = find_boundaries(cell[y0:y1, x0:x1], mode="inner")
    rgb[cb] = (255, 60, 220)
    rgb[nb] = (40, 230, 255)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_path)
