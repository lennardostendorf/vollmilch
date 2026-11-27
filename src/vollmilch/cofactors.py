"""Choosing asinh cofactors: per-sample suggestions from the intensity distributions, an HTML
viewer with ridge plots and sliders, and the CSV that `transform.cofactors` reads.

Suggestion: the cofactor is put at the upper edge of the background (negative) population,
background mode + 3 x background SD. Background and noise then stay in the near-linear part of
asinh, compressed around 0, and signal above it is log-scaled. Mode and SD come from the
histogram peak; the SD from its half width at half maximum on the low side, which positive
cells do not reach. On standard data this gives the usual values (about 5 for CyTOF counts,
1-2 for IMC counts) and a value just above the camera offset for raw fluorescence.

Background-subtracted images (background thresholded to exactly 0, e.g. the standard MELC
preprocessing) have no background noise left to measure. They are recognised by a pile-up of exact
zeros: most pixels, or cells whose every pixel is 0, which noise never produces. If that holds for
most markers, the image is treated as background-subtracted and every marker gets
  p1-cells   (default) the 1st percentile of the positive (> 0) cell values, or
  p1-pixels  the 1st percentile of the positive pixel values (also before segmentation).
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import __version__
from .embed import cofactor as config_cofactor
from .report import Blob

log = logging.getLogger("vollmilch")

K_SD = 3.0      # cofactor = background mode + K_SD x background SD
REMOVED = {"pixels": 0.6, "cells": 0.002}  # share piled up at the minimum that means: background removed
# (noise never makes a whole cell exactly 0, while a clipped noisy background can be ~half the pixels)
N_FINE = 2048   # histogram bins per sample and marker stored in the viewer
SMOOTH = 1.5    # histogram smoothing (bins) for the peak search
HWHM_PER_SD = np.sqrt(2 * np.log(2))
SUBTRACTED_RULES = ("p1-cells", "p1-pixels")  # cofactor rules for background-subtracted markers
P1_MIN_N = 20         # positive values needed for a 1st percentile


def sig2(x: float) -> float:
    """Round to 2 significant digits (180, 1.3, 0.042): readable values for a config."""
    return float(f"{x:.2g}")


# ---------------------------------------------------------------------------- suggestion
def _half_width(hs: np.ndarray, i: int):
    """Half width at half maximum of the peak at i, in bins: (left, right); None if the
    curve does not drop below half maximum on that side."""
    half = hs[i] / 2
    lw = rw = None
    below = np.flatnonzero(hs[:i] < half)
    if below.size:
        j = below[-1]
        lw = i - (j + (half - hs[j]) / (hs[j + 1] - hs[j]))
    below = np.flatnonzero(hs[i + 1:] < half)
    if below.size:
        j = i + 1 + below[0]
        rw = (j - 1 + (hs[j - 1] - half) / (hs[j - 1] - hs[j])) - i
    return lw, rw


def _peak(v: np.ndarray, lo: float, hi: float, nbins: int, integer: bool):
    """Mode and SD of the highest histogram peak in [lo, hi]. Returns (mode, sd, bin width)."""
    if integer:  # bins on whole numbers, so integer data does not give a comb
        lo, hi = np.floor(lo) - 0.5, np.ceil(hi) + 0.5
        width = max(1.0, float(np.ceil((hi - lo) / nbins)))
        nbins = max(1, int(np.ceil((hi - lo) / width)))
        hi = lo + nbins * width
    else:
        width = (hi - lo) / nbins
    h, _ = np.histogram(v, nbins, (lo, hi))
    smooth = 0.0 if integer and width == 1 else SMOOTH  # one bin per value: nothing to smooth
    hs = ndi.gaussian_filter1d(h.astype(np.float64), smooth, mode="nearest") if smooth else h.astype(np.float64)
    i = int(np.argmax(hs))
    lw, rw = _half_width(hs, i)
    hw = lw if lw is not None else rw if rw is not None else 1.0
    # remove the widening by the smoothing and the binning
    sd_bins = np.sqrt(max((hw / HWHM_PER_SD) ** 2 - smooth**2 - 1 / 12, 1 / 12))
    return lo + (i + 0.5) * width, sd_bins * width, width


def pileup(v: np.ndarray, end: str = "min") -> float:
    """Share of values sitting exactly at the minimum (maximum), if they pile up there: more
    than 5x as many as at the next value. Poisson-like counts (0 > 1 > 2 ...) are no pile-up."""
    x = v.min() if end == "min" else v.max()
    at = np.count_nonzero(v == x)
    rest = v[v > x] if end == "min" else v[v < x]
    if not rest.size:
        return 0.0
    nxt = rest.min() if end == "min" else rest.max()
    return at / v.size if at > 5 * np.count_nonzero(rest == nxt) else 0.0


def _drop_pileup(v: np.ndarray, end: str, min_frac: float) -> np.ndarray:
    if pileup(v, end) < min_frac:
        return v
    rest = v[v > v.min()] if end == "min" else v[v < v.max()]
    return rest if rest.size >= 50 else v


def _drop_clipped(v: np.ndarray) -> np.ndarray:
    """Values piled up at the minimum (background subtracted and clipped at 0) or at the
    maximum (saturation) are not the shape of the background; leave them out. A clipped
    background peak is then measured on its upper side."""
    return _drop_pileup(_drop_pileup(v, "min", 0.01), "max", 0.005)


def background(v: np.ndarray):
    """(mode, sd) of the background peak: a coarse pass over the whole range, then passes
    around the peak with bins of about SD / 4 until the SD settles. Bins scale with the peak,
    not the data range, so a few thousand cells work as well as millions of pixels."""
    v = v[np.isfinite(v)]
    if v.size < 50:
        return np.nan, np.nan
    v = _drop_clipped(v)
    integer = bool(np.all(v[:10000] == np.round(v[:10000])))
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo:
        return float(lo), 1.0 if integer else max(abs(float(lo)), 1.0) * 1e-3
    vmin = float(v.min())
    m, s, w = _peak(v, lo, hi, 1024, integer)
    for _ in range(6):
        scale = max(s, w)  # an unresolved peak (s < bin width) is zoomed into 4x finer bins
        lo2, hi2 = max(vmin, m - 8 * scale), m + 8 * scale
        m, s_new, w = _peak(v, lo2, hi2, max(16, int(np.ceil((hi2 - lo2) / (scale / 4)))), integer)
        done = s_new > w and abs(s_new - s) <= 0.05 * s
        s = s_new
        if done or (integer and w == 1):
            break
    return float(m), float(s)


def p1_positive(v: np.ndarray) -> float:
    """1st percentile of the positive (> 0) values; NaN with fewer than P1_MIN_N of them."""
    pos = v[v > 0]
    return float(np.percentile(pos, 1)) if pos.size >= P1_MIN_N else np.nan


def suggest(v: np.ndarray, level: str = "pixels") -> dict:
    """Suggested cofactor from the background: mode + 3 SD (method "background"). frac_below is
    the fraction of values clearly below the peak (< mode - 3 SD): background is the lowest
    population, so a large value means the main peak is something else (e.g. a marker present on
    most cells). zero_frac is the share piled up at the minimum; at REMOVED[level] or more the
    background was removed: no estimate here (method "none"), run_cofactors applies a p1 rule."""
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    zero_frac = pileup(v) if v.size else 0.0
    removed = zero_frac >= REMOVED[level]
    out = {"cofactor": np.nan, "method": "none", "background": np.nan, "noise_sd": np.nan,
           "frac_above": np.nan, "frac_below": np.nan, "zero_frac": float(zero_frac),
           "background_removed": bool(removed), "n_values": int(v.size)}
    if removed:
        return out
    mode, sd = background(v)
    if not np.isfinite(mode):
        return out
    integer = bool(np.all(v[:10000] == np.round(v[:10000])))
    floor = 1.0 if integer else 1e-6 * max(float(np.abs(v).max()), 1e-12)
    c = sig2(max(max(mode, 0.0) + K_SD * sd, floor))
    out.update(cofactor=c, method="background", background=float(mode), noise_sd=float(sd),
               frac_above=float(np.mean(v > c)), frac_below=float(np.mean(v < mode - K_SD * sd)))
    return out


# ---------------------------------------------------------------------------- values
def marker_channels(cfg, source) -> list[str]:
    """The channels that get features (and so a cofactor): features.channels minus exclude."""
    fc = cfg["features"]
    chans = source.channel_names if fc["channels"] in ("all", None) else list(fc["channels"])
    excl = set(fc.get("exclude") or [])
    return [source.channel_names[source.channel_index(c)] for c in chans if c not in excl]


def pixel_windows(region, max_pixels: int, rng, tile: int = 256):
    """Random windows inside the region, together holding about max_pixels region pixels.
    Returns [(bbox in image pixels, mask of the window)]."""
    mask = region.mask()
    H, W = mask.shape
    y0, _, x0, _ = region.bbox
    if mask.sum() <= max_pixels:
        return [((y0, y0 + H, x0, x0 + W), mask)]
    cand = []
    for ty in range(0, H, tile):
        for tx in range(0, W, tile):
            m = mask[ty:ty + tile, tx:tx + tile]
            if m.mean() > 0.25:
                cand.append((ty, tx, int(m.sum())))
    if not cand:  # thin region: windows with any region pixels
        cand = [(ty, tx, int(mask[ty:ty + tile, tx:tx + tile].sum()))
                for ty in range(0, H, tile) for tx in range(0, W, tile) if mask[ty:ty + tile, tx:tx + tile].any()]
    out, total = [], 0
    for k in rng.permutation(len(cand)):
        ty, tx, n = cand[k]
        ty1, tx1 = min(ty + tile, H), min(tx + tile, W)
        out.append(((y0 + ty, y0 + ty1, x0 + tx, x0 + tx1), mask[ty:ty1, tx:tx1]))
        total += n
        if total >= max_pixels:
            break
    return out


def cell_column(df: pd.DataFrame, cfg, marker: str) -> str:
    """The cell-table column the embedding transforms (embedding.compartment / stat)."""
    ec = cfg["embedding"]
    comp, stat = ec.get("compartment", "cell"), ec.get("stat", "mean")
    for c in (comp, "nucleus"):  # nuclei mode has no separate cell compartment
        if f"{marker}:{c}:{stat}" in df.columns:
            return f"{marker}:{c}:{stat}"
    raise KeyError(f"no column {marker}:{comp}:{stat} in the cell table")


def _hist(v: np.ndarray, c0: float):
    """Fine histogram on a grid uniform in asinh(x / c0): log-spaced for |x| >> c0 and linear
    around 0, so the viewer can re-bin it for any cofactor >= about c0."""
    u = np.arcsinh(v / c0)
    u0, u1 = float(u.min()), float(u.max())
    if u1 <= u0:
        u0, u1 = u0 - 0.5, u1 + 0.5
    h, _ = np.histogram(u, N_FINE, (u0, u1))
    return h.astype(np.uint32), u0, u1


# ---------------------------------------------------------------------------- command
def run_cofactors(cfg, level="auto", samples=None, max_pixels=2_000_000, out_path=None, subtracted="p1-cells") -> Path:
    from .pipeline import completed_samples, load_cells, sample_seed
    from .regions import get_regions
    from .sources import open_source

    if subtracted not in SUBTRACTED_RULES:
        raise ValueError(f"subtracted must be one of {SUBTRACTED_RULES}")
    t0 = time.time()
    source = open_source(cfg)
    markers = marker_channels(cfg, source)
    if level == "auto":  # cells once every sample is segmented: those are the values transformed
        done = set(completed_samples(cfg))
        want = [r.sample for r in get_regions(cfg, source) if not samples or r.sample in set(samples) or r.name in set(samples)]
        level = "cells" if want and all(w in done for w in want) else "pixels"
        log.info("cofactors from %s (%s)", level, "all samples segmented" if level == "cells" else
                 "not all samples segmented yet; once they are, run again for cell-level suggestions")
    if level == "pixels":
        regions = get_regions(cfg, source)
        if samples:
            want = set(samples)
            regions = [r for r in regions if r.sample in want or r.name in want]
        names = [r.sample for r in regions]
        df = None
    elif level == "cells":
        regions = None
        names = [s for s in completed_samples(cfg) if not samples or s in set(samples)]
        df = load_cells(cfg, names) if names else None
    else:
        raise ValueError("level must be auto, pixels or cells")
    if not names:
        raise ValueError("no samples" + ("" if level == "pixels" else "; run `vollmilch segment` first or use --from pixels"))
    log.info("cofactors from %s: %d samples, %d markers", level, len(names), len(markers))
    read_px = level == "pixels" or subtracted == "p1-pixels"
    if read_px and regions is None:
        by_sample = {r.sample: r for r in get_regions(cfg, source)}
        regions = [by_sample[s] for s in names]

    S, Mk = len(names), len(markers)
    stats = [[None] * S for _ in range(Mk)]
    hists = np.zeros((Mk, S, N_FINE), np.uint32)
    grids = np.full((Mk, S, 3), np.nan)          # c0, u0, u1
    ranges = np.full((Mk, S, 2), np.nan)         # 0.1 and 99.9 percentile
    for si, s in enumerate(names):
        if read_px:
            r = regions[si]
            wins = pixel_windows(r, max_pixels, np.random.default_rng(sample_seed(s, 2)))
        if level == "cells":
            sel = (df["sample"] == s).to_numpy()
        for mi, m in enumerate(markers):
            px = np.concatenate([source.read(r.image, m, bb)[wm] for bb, wm in wins]).astype(np.float64) if read_px else None
            v = px if level == "pixels" else df[cell_column(df, cfg, m)].to_numpy(np.float64)[sel]
            v = v[np.isfinite(v)]
            st = suggest(v, level)
            st["p1_cells"] = p1_positive(v) if level == "cells" else np.nan
            st["p1_pixels"] = p1_positive(px) if px is not None else np.nan
            if px is not None and level == "cells" and pileup(px) >= REMOVED["pixels"]:
                st["background_removed"] = True  # e.g. DAPI: no zero cells, but most pixels 0
            stats[mi][si] = st
            if v.size:
                # grid fine enough for the smallest cofactor the slider is likely to reach
                c0 = min((x for x in (st["cofactor"], st["p1_cells"], st["p1_pixels"], config_cofactor(cfg, m))
                          if np.isfinite(x) and x > 0), default=max(float(np.abs(v).max()), 1.0)) / 30
                hists[mi, si], u0, u1 = _hist(v, c0)
                grids[mi, si] = (c0, u0, u1)
                ranges[mi, si] = np.percentile(v, [0.1, 99.9])

    # background-subtracted markers, or all markers once most of them are: the chosen p1 rule
    n_removed = sum(any(st["background_removed"] for st in stats[mi]) for mi in range(Mk))
    is_subtracted = n_removed >= Mk / 2
    key = "p1_cells" if subtracted == "p1-cells" else "p1_pixels"
    for mi in range(Mk):
        for st in stats[mi]:
            if st["background_removed"] or is_subtracted:
                c = st[key]
                st.update(cofactor=sig2(c) if np.isfinite(c) and c > 0 else np.nan,
                          method=subtracted if np.isfinite(c) and c > 0 else "none",
                          background=np.nan, noise_sd=np.nan, frac_above=np.nan, frac_below=np.nan)
    for si, s in enumerate(names):
        log.info("  %s: %s", s, ", ".join(f"{m} {stats[mi][si]['cofactor']:g}" if np.isfinite(stats[mi][si]["cofactor"])
                                          else f"{m} -" for mi, m in enumerate(markers)))
    if is_subtracted or n_removed:
        what = "the image is background-subtracted" if is_subtracted else f"{n_removed} markers are background-subtracted"
        if subtracted == "p1-cells" and level == "pixels":
            log.warning("%s: the suggestion is the 1st percentile of the positive cell values, so segment first and "
                        "run `vollmilch cofactors` again (or use --subtracted p1-pixels)", what)
        else:
            log.info("%s: cofactor = 1st percentile of the positive %s values", what,
                     "cell" if subtracted == "p1-cells" else "pixel")
    source.close()

    out_dir = cfg.out / "cofactors"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample = pd.DataFrame([{"marker": m, "sample": s, **stats[mi][si]} for mi, m in enumerate(markers) for si, s in enumerate(names)])
    per_sample.to_csv(out_dir / "per_sample.csv", index=False)

    cof = cfg["transform"].get("cofactors") or {}
    mmeta, rows = [], []
    for mi, m in enumerate(markers):
        cs = np.array([st["cofactor"] for st in stats[mi]], np.float64)
        fb = np.array([st["frac_below"] for st in stats[mi]], np.float64)
        cur = config_cofactor(cfg, m)
        zf = np.array([st["zero_frac"] for st in stats[mi]], np.float64)
        sug = sig2(float(np.nanmedian(cs))) if np.isfinite(cs).any() else None
        lo, hi = np.nanmin(ranges[mi, :, 0]), np.nanmax(ranges[mi, :, 1])
        cands = [x for x in (sug, cur, *cs) if x is not None and np.isfinite(x) and x > 0]
        methods = sorted({st["method"] for st in stats[mi]} - {"none"})
        rows.append({"marker": m, "cofactor": sug, "method": "+".join(methods) or "none", "config": cur,
                     "samples_min": np.nanmin(cs) if np.isfinite(cs).any() else np.nan,
                     "samples_max": np.nanmax(cs) if np.isfinite(cs).any() else np.nan,
                     "frac_below_peak": float(np.nanmedian(fb)) if np.isfinite(fb).any() else np.nan,
                     "zero_frac": float(np.median(zf)),
                     "background_removed": int(sum(st["background_removed"] for st in stats[mi]))})
        mmeta.append({
            "name": m, "current": cur, "suggested": sug, "in_config": m in cof,
            "slider": [min(cands) / 30, max(cands) * 30],
            "range": [float(lo), float(hi)] if np.isfinite(lo) else [0.0, 1.0],
            "per_sample": [{k: (None if not np.isfinite(x) else round(x, 4)) if isinstance(x, float) else x
                            for k, x in st.items()} for st in stats[mi]],
            "grid": [[None if not np.isfinite(x) else float(x) for x in g] for g in grids[mi]],
        })
    pd.DataFrame(rows).to_csv(out_dir / "suggested.csv", index=False)

    blob = Blob()
    for mi in range(Mk):
        blob.add(f"hist:{mi}", hists[mi].ravel())
    cf_src = cfg["transform"].get("_cofactors_file")
    manifest = {
        "title": f"{cfg.get('project') or 'vollmilch'} · cofactors",
        "report_id": hashlib.sha1(f"{cfg.out.resolve()}|{cfg.get('project')}|cofactors".encode()).hexdigest()[:12],
        "created": time.strftime("%Y-%m-%d %H:%M"), "version": __version__, "level": level,
        "max_pixels": max_pixels if level == "pixels" else None,
        "samples": names, "markers": mmeta, "n_fine": N_FINE, "k_sd": K_SD, "subtracted_rule": subtracted, "is_subtracted": bool(is_subtracted), "blob": blob.index,
        "default_cofactor": float(cfg["transform"].get("cofactor", 5)),
        "config_name": Path(cfg["_config_path"]).name,
        "cofactors_file": Path(cf_src).name if cf_src else None,
    }
    tpl = resources.files("vollmilch").joinpath("viewer/cofactors.html").read_text(encoding="utf-8")
    logo = resources.files("vollmilch").joinpath("assets/logo.svg").read_text(encoding="utf-8")
    html = (tpl.replace("__TITLE__", manifest["title"].replace("<", "&lt;"))
               .replace("/*__MANIFEST__*/null", json.dumps(manifest, allow_nan=False).replace("</", "<\\/"))
               .replace("__DATA_B64__", blob.encode())
               .replace("__LOGO_SVG__", logo)
               .replace("__FAVICON__", "data:image/svg+xml;base64," + base64.b64encode(logo.encode()).decode()))
    out_path = Path(out_path) if out_path else out_dir / "cofactors.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    log.info("cofactor viewer written: %s (%.1f MB, %.0f s)", out_path, out_path.stat().st_size / 1e6, time.time() - t0)
    return out_path
