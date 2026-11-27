"""Builds the self-contained HTML gating viewer."""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import logging
import time
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import __version__
from .gating import load_embeddings, load_gates, viewer_features
from .pipeline import completed_samples, load_cells, sample_meta, sample_seed

log = logging.getLogger("vollmilch")

PALETTE = ["#00e05a", "#ff3b3b", "#ff3df2", "#00d5ff", "#ffd500", "#ff8c00", "#ffffff", "#9b7bff", "#7dff7d", "#ff9eb5"]
NUCLEAR = {"dapi", "hoechst", "dna", "dna1", "dna2", "nuclei", "nucleus"}


class Blob:
    def __init__(self):
        self.parts, self.index, self.off = [], {}, 0

    def add(self, name, arr: np.ndarray):
        arr = np.ascontiguousarray(arr)
        pad = (-self.off) % 8
        if pad:
            self.parts.append(b"\0" * pad)
            self.off += pad
        b = arr.tobytes()
        self.index[name] = {"offset": self.off, "dtype": arr.dtype.name, "length": int(arr.size)}
        self.parts.append(b)
        self.off += len(b)

    def encode(self) -> str:
        raw = b"".join(self.parts)
        return base64.b64encode(gzip.compress(raw, compresslevel=6, mtime=0)).decode("ascii")


def _choose_rows(df: pd.DataFrame, thumb_ids: dict, cap: int, cap_total: int) -> np.ndarray:
    rows = []
    n_samples = df["sample"].nunique()
    per = min(cap, max(1000, cap_total // max(1, n_samples)))
    for s, idx in df.groupby("sample", sort=False).indices.items():
        idx = np.asarray(idx)
        if len(idx) <= per:
            rows.append(idx)
            continue
        oid = df["object_id"].to_numpy()[idx]
        has_t = np.isin(oid, thumb_ids.get(s, np.zeros(0, int)))
        must = idx[has_t]
        rest = idx[~has_t]
        rng = np.random.default_rng(sample_seed(s, 1))
        k = max(0, per - len(must))
        rows.append(np.concatenate([must, rng.choice(rest, min(k, len(rest)), replace=False)]))
    return np.sort(np.concatenate(rows))


def build_report(cfg, out_path: Path | None = None, gates_path=None, title=None) -> Path:
    t0 = time.time()
    samples = completed_samples(cfg)
    df = load_cells(cfg, samples)
    embs = load_embeddings(cfg)
    ids = set(df["cell_id"])
    for name, e in embs.items():
        missing = len(ids - set(e["cell_id"]))
        if missing and name == "all":
            log.warning("embedding '%s' lacks %d of %d cells (segmentation changed?); re-run `vollmilch embed`", name, missing, len(ids))
    rc = cfg["report"]
    tc = cfg["thumbnails"]

    # thumbnails per sample
    thumbs = {}
    for s in samples:
        p = cfg.sample_dir(s) / "thumbs.npz"
        if p.exists():
            z = np.load(p, allow_pickle=False)
            thumbs[s] = z
    thumb_ids = {s: z["ids"] for s, z in thumbs.items()}
    rows = _choose_rows(df, thumb_ids, int(rc.get("max_cells_per_sample", 150000)), int(rc.get("max_cells_total", 600000)))
    d = df.iloc[rows].reset_index(drop=True)
    n = len(d)
    log.info("report: %d of %d cells", n, len(df))
    vals, desc, cats = viewer_features(d, cfg, embs)

    blob = Blob()
    for c in desc:
        blob.add("num:" + c["key"], vals[c["key"]].to_numpy(np.float32))
    cat_meta = []
    for k, v in cats.items():
        levels = sorted(set(v), key=lambda x: (len(x) if x.isdigit() else 0, x)) if k.startswith("leiden") else list(dict.fromkeys(v))
        if not k.startswith("leiden") and k != "sample":
            levels = sorted(levels)
        lut = {l: i for i, l in enumerate(levels)}
        codes = np.array([lut[x] for x in v], np.uint16)
        blob.add("cat:" + k, codes)
        cat_meta.append({"key": k, "levels": levels})
    blob.add("object_id", d["object_id"].to_numpy(np.uint32))

    # thumbnails -> uint8 (sqrt-encoded), channel order from config
    t_chans = list(tc.get("channels") or [])
    thumb_meta = None
    tblob = None
    if thumbs and t_chans:
        row_of = {(s, int(o)): i for i, (s, o) in enumerate(zip(d["sample"], d["object_id"]))}
        sel = []  # (sample, index in npz, row)
        for s, z in thumbs.items():
            chs = list(z["channels"])
            if chs != t_chans:
                log.warning("%s: thumbnail channels %s differ from config %s; re-run segment with --reuse-segmentation", s, chs, t_chans)
                continue
            for j, o in enumerate(z["ids"]):
                r = row_of.get((s, int(o)))
                if r is not None:
                    sel.append((s, j, r))
        cap_t = int(rc.get("max_thumbnails_total", 40000))
        if len(sel) > cap_t:
            rng = np.random.default_rng(7)
            keep = np.sort(rng.choice(len(sel), cap_t, replace=False))
            sel = [sel[i] for i in keep]
        if sel:
            half = int(next(iter(thumbs.values()))["half"])
            size = 2 * half
            C = len(t_chans)
            ranges = []
            for ci in range(C):
                samp = np.concatenate([thumbs[s][f"img_{ci}"].ravel()[::7] for s in thumbs if list(thumbs[s]["channels"]) == t_chans]).astype(np.float64)
                lo, hi = np.percentile(samp, [0.5, 99.9])
                ranges.append((float(lo), float(max(hi, lo + 1e-6))))
            T = np.zeros((len(sel), C, size, size), np.uint8)
            O = np.zeros((len(sel), size, size), np.uint8)
            trow = np.zeros(len(sel), np.int32)
            by_s = {}
            for k, (s, j, r) in enumerate(sel):
                by_s.setdefault(s, []).append((k, j, r))
            for s, items in by_s.items():
                z = thumbs[s]
                ks = np.array([i[0] for i in items])
                js = np.array([i[1] for i in items])
                trow[ks] = [i[2] for i in items]
                O[ks] = z["outlines"][js]
                for ci in range(C):
                    lo, hi = ranges[ci]
                    a = z[f"img_{ci}"][js].astype(np.float32)
                    T[ks, ci] = np.round(255 * np.sqrt(np.clip((a - lo) / (hi - lo), 0, 1))).astype(np.uint8)
            o = np.argsort(trow)
            T, O, trow = T[o], O[o], trow[o]
            tblob = Blob()
            atlases = []
            q = int(rc.get("thumbnail_jpeg_quality", 90))
            per, ncol = 1024, 32
            for ci in range(C):
                for start in range(0, len(T), per):
                    block = T[start:start + per, ci]
                    k = len(block)
                    nrow = int(np.ceil(k / ncol))
                    pad = np.zeros((nrow * ncol, size, size), np.uint8)
                    pad[:k] = block
                    atlas = pad.reshape(nrow, ncol, size, size).transpose(0, 2, 1, 3).reshape(nrow * size, ncol * size)
                    bio = io.BytesIO()
                    Image.fromarray(atlas).save(bio, "JPEG", quality=q) if q < 100 else Image.fromarray(atlas).save(bio, "PNG")
                    key = f"atlas:{ci}:{start}"
                    tblob.add(key, np.frombuffer(bio.getvalue(), np.uint8))
                    atlases.append({"key": key, "channel": ci, "start": start, "count": k, "cols": ncol,
                                    "mime": "image/jpeg" if q < 100 else "image/png"})
            tblob.add("outlines", O)
            tblob.add("rows", trow)
            colors = list(tc.get("colors") or [])
            pal = iter(PALETTE)
            for i, c in enumerate(t_chans):
                if i >= len(colors) or not colors[i]:
                    colors = colors[:i] + ["#3d6bff" if c.lower() in NUCLEAR else next(pal)] + colors[i + 1:]
            thumb_meta = {"channels": t_chans, "colors": colors[: len(t_chans)], "size": size, "n": len(sel), "atlases": atlases,
                          "size_um": size * float(sample_meta(cfg, samples[0])["pixel_size_um"])}
            log.info("report: %d thumbnails (%d channels, %d px)", len(sel), C, size)

    smeta = []
    for s in samples:
        m = sample_meta(cfg, s)
        smeta.append({"sample": s, "image": m["image"], "region": m["name"], "n_total": int((df["sample"] == s).sum()),
                      "n_shown": int((d["sample"] == s).sum()), "image_path": m.get("image_path"),
                      "pixel_size_um": m["pixel_size_um"]})
    emb_meta = []
    for name in embs:
        p = cfg.out / "embeddings" / f"{name}.json"
        if p.exists():
            emb_meta.append(json.loads(p.read_text()))
    gates = load_gates(gates_path) if gates_path else None
    rid = hashlib.sha1(f"{cfg.out.resolve()}|{cfg.get('project')}".encode()).hexdigest()[:12]
    manifest = {
        "title": title or rc.get("title") or cfg.get("project") or "vollmilch",
        "report_id": rid, "created": time.strftime("%Y-%m-%d %H:%M"), "version": __version__,
        "n": n, "n_total": int(len(df)), "columns": desc, "categories": cat_meta, "samples": smeta,
        "embeddings": emb_meta, "thumbs": thumb_meta, "blob": blob.index,
        "thumb_blob": tblob.index if tblob else None, "gates": gates,
        "segmentation_mode": cfg["segmentation"]["mode"],
    }
    tpl = resources.files("vollmilch").joinpath("viewer/viewer.html").read_text(encoding="utf-8")
    html = (tpl.replace("__TITLE__", manifest["title"].replace("<", "&lt;"))
               .replace("/*__MANIFEST__*/null", json.dumps(manifest, allow_nan=False).replace("</", "<\\/"))
               .replace("__DATA_B64__", blob.encode())
               .replace("__THUMBS_B64__", tblob.encode() if tblob else ""))
    logo = resources.files("vollmilch").joinpath("assets/logo.svg").read_text(encoding="utf-8")
    html = (html.replace("__LOGO_SVG__", logo)
                .replace("__FAVICON__", "data:image/svg+xml;base64," + base64.b64encode(logo.encode()).decode()))
    out_path = Path(out_path) if out_path else cfg.out / "report.html"
    out_path.write_text(html, encoding="utf-8")
    log.info("report written: %s (%.1f MB, %.0f s)", out_path, out_path.stat().st_size / 1e6, time.time() - t0)
    return out_path
