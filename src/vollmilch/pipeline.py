"""Per-sample processing (segmentation, features, thumbnails) and loading of the results."""
from __future__ import annotations

import json
import logging
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from .features import extract_features, qc_overlay
from .regions import get_regions
from .segmentation import make_backend, segment_region
from .sources import PngFolderSource, open_source

log = logging.getLogger("vollmilch")


def setup_logging(out: Path | None = None, verbose=False):
    lg = logging.getLogger("vollmilch")
    lg.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in lg.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        lg.addHandler(h)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(out / "vollmilch.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        lg.addHandler(fh)


def sample_seed(name: str, seed: int = 0) -> int:
    return (zlib.crc32(name.encode()) + seed) & 0xFFFFFFFF


def _write_labels(path: Path, lab: np.ndarray):
    tifffile.imwrite(path, lab.astype(np.uint32), compression="zlib", tile=(512, 512) if min(lab.shape) >= 512 else None)


def read_labels(path: Path) -> np.ndarray:
    return tifffile.imread(path)


def process_sample(source, region, cfg, backend=None, force=False, reuse_segmentation=False):
    d = cfg.sample_dir(region.sample)
    d.mkdir(parents=True, exist_ok=True)
    meta_p = d / "meta.json"
    if not force and not reuse_segmentation and meta_p.exists() and json.loads(meta_p.read_text()).get("complete"):
        log.info("%s: done already (use --force to redo)", region.sample)
        return backend
    t0 = time.time()
    log.info("%s: region %s of image %s, %d x %d px", region.sample, region.name, region.image, region.shape[1], region.shape[0])
    nuc_p, cell_p = d / "nuclei_labels.tif", d / "cell_labels.tif"
    if reuse_segmentation and nuc_p.exists() and cell_p.exists():
        nuc, cell = read_labels(nuc_p), read_labels(cell_p)
        log.info("  reusing segmentation (%d objects)", int(cell.max()))
    else:
        if backend is None:
            backend = make_backend(cfg["segmentation"], source.pixel_size_um)
        nuc, cell = segment_region(source, region, cfg, backend)
        _write_labels(nuc_p, nuc)
        _write_labels(cell_p, cell)
    n = int(cell.max())
    log.info("  %d objects; extracting features", n)
    rng = np.random.default_rng(sample_seed(region.sample, cfg["embedding"].get("seed", 0)))
    df, thumbs = extract_features(source, region, nuc, cell, cfg, rng)
    df.insert(0, "cell_id", [f"{region.sample}:{i}" for i in df["object_id"]])
    df.insert(1, "sample", region.sample)
    df.insert(2, "image", region.image)
    df.to_parquet(d / "cells.parquet", index=False)
    if len(thumbs["ids"]):
        np.savez_compressed(
            d / "thumbs.npz", ids=thumbs["ids"], channels=np.array(thumbs["channels"]),
            outlines=thumbs.get("outlines"), half=thumbs["half"],
            **{f"img_{i}": thumbs["imgs"][c] for i, c in enumerate(thumbs["channels"])},
        )
    try:
        qc_overlay(source, region, nuc, cell, cfg, cfg.out / "qc" / f"{region.sample}_segmentation.png")
    except Exception as e:  # QC image is a convenience
        log.warning("  QC overlay failed: %s", e)
    meta = region.to_json()
    meta.update({
        "complete": True, "n_objects": n, "pixel_size_um": source.pixel_size_um,
        "segmentation_mode": cfg["segmentation"]["mode"], "backend": cfg["segmentation"]["backend"],
        "image_path": source.image(region.image).path, "seconds": round(time.time() - t0, 1),
    })
    meta_p.write_text(json.dumps(meta, indent=1))
    log.info("%s: %d cells in %.0f s", region.sample, n, time.time() - t0)
    return backend


def run_segmentation(cfg, samples=None, force=False, reuse_segmentation=False):
    source = open_source(cfg)
    regions = get_regions(cfg, source)
    if samples:
        want = set(samples)
        regions = [r for r in regions if r.sample in want or r.name in want]
        if not regions:
            raise ValueError(f"none of {sorted(want)} found")
    if isinstance(source, PngFolderSource):
        for im in source.images:
            p = Path(im.path)
            if not p.exists():
                source.write_ome_tiff(im.name, p)
                log.info("wrote %s (open this in QuPath)", p)
    (cfg.out).mkdir(parents=True, exist_ok=True)
    (cfg.out / "regions.json").write_text(json.dumps([r.to_json() for r in get_regions(cfg, source)], indent=1))
    log.info("%d samples, %d channels, %.4f µm/px", len(regions), len(source.channel_names), source.pixel_size_um)
    backend = None
    for r in regions:
        backend = process_sample(source, r, cfg, backend, force=force, reuse_segmentation=reuse_segmentation)
    source.close()


# ---------------------------------------------------------------------------- loading
def completed_samples(cfg) -> list[str]:
    root = cfg.out / "samples"
    out = []
    if root.exists():
        for d in sorted(root.iterdir()):
            m = d / "meta.json"
            if m.exists() and json.loads(m.read_text()).get("complete") and (d / "cells.parquet").exists():
                out.append(d.name)
    return out


def load_samples_table(cfg) -> pd.DataFrame | None:
    p = cfg.get("samples_table")
    if not p:
        return None
    t = pd.read_csv(p, sep=None, engine="python", dtype=str)
    if "sample" not in t.columns:
        raise ValueError(f"{p} needs a 'sample' column")
    return t


def load_cells(cfg, samples=None) -> pd.DataFrame:
    names = samples or completed_samples(cfg)
    if not names:
        raise FileNotFoundError(f"no finished samples in {cfg.out / 'samples'}; run `vollmilch segment` first")
    df = pd.concat([pd.read_parquet(cfg.sample_dir(s) / "cells.parquet") for s in names], ignore_index=True)
    st = load_samples_table(cfg)
    if st is not None:
        extra = [c for c in st.columns if c != "sample" and c not in df.columns]
        missing = set(df["sample"]) - set(st["sample"])
        if missing:
            log.warning("samples missing from samples_table: %s", sorted(missing))
        df = df.merge(st[["sample", *extra]], on="sample", how="left")
    return df


def sample_meta(cfg, sample) -> dict:
    return json.loads((cfg.sample_dir(sample) / "meta.json").read_text())
