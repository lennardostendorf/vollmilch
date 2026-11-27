"""Synthetic multiplex data for tests: a PerkinElmer-style .qptiff with two tissue pieces
(+ QuPath-like GeoJSON outlines) and a folder of PNG FOVs (one PNG per channel)."""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.draw import ellipse

CHANNELS = ["DAPI", "CD45", "CD3", "CD20", "CD68", "PanCK", "NaKATP", "Ki67"]
TYPES = {  # name: (prob, markers on membrane/cytoplasm, nucleus radius, cell extra radius)
    "T": (0.30, {"CD45": 1.0, "CD3": 1.0}, 5.5, 3),
    "B": (0.15, {"CD45": 1.0, "CD20": 1.0}, 6, 3),
    "Mac": (0.15, {"CD45": 0.6, "CD68": 1.0}, 7, 7),
    "Epi": (0.30, {"PanCK": 1.0, "NaKATP": 1.0}, 7.5, 7),
    "Str": (0.10, {"NaKATP": 0.5}, 6, 4),
}


def paint(shape, centers, rng, gain=1.0, region_mask=None):
    H, W = shape
    nuc = np.zeros(shape, np.int32)
    cell = np.zeros(shape, np.int32)
    imgs = {c: np.zeros(shape, np.float32) for c in CHANNELS}
    types = list(TYPES)
    probs = np.array([TYPES[t][0] for t in types])
    truth = []
    for k, (cy, cx) in enumerate(centers, 1):
        t = types[rng.choice(len(types), p=probs / probs.sum())]
        _, mk, rn, extra = TYPES[t]
        rn = rn * rng.uniform(0.85, 1.15)
        ry, rx = rn * rng.uniform(0.8, 1.2), rn * rng.uniform(0.8, 1.2)
        rot = rng.uniform(0, np.pi)
        cr, cc = ellipse(cy, cx, ry + extra, rx + extra, shape=shape, rotation=rot)
        free = cell[cr, cc] == 0
        cell[cr[free], cc[free]] = k
        nr, nc = ellipse(cy, cx, ry, rx, shape=shape, rotation=rot)
        nuc[nr, nc] = k
        cell[nr, nc] = k
        truth.append((k, t, cy, cx))
        ki = rng.random() < 0.15
        dapi = 3000 * rng.uniform(0.7, 1.3)
        imgs["DAPI"][nr, nc] = dapi * (1 + 0.35 * rng.standard_normal(len(nr)))
        if ki:
            imgs["Ki67"][nr, nc] = 2500 * gain
        cyto = cell == k
        cyto[nr, nc] = False
        for m, v in mk.items():
            lvl = v * rng.uniform(0.5, 1.5) * 2000 * gain
            imgs[m][cyto] += lvl
    for c in CHANNELS:
        im = ndi.gaussian_filter(imgs[c], 1.0) + 80 + 40 * rng.random(shape).astype(np.float32)
        im = rng.poisson(np.clip(im, 0, None)).astype(np.float32)
        if region_mask is not None:
            im[~region_mask] = rng.poisson(60, (~region_mask).sum())
        imgs[c] = np.clip(im, 0, 65535).astype(np.uint16)
    return imgs, nuc, cell, truth


def centers_in(mask, n, mind, rng):
    ys, xs = np.nonzero(mask)
    out = []
    tries = 0
    pts = np.zeros((0, 2))
    while len(out) < n and tries < n * 30:
        tries += 1
        i = rng.integers(len(ys))
        p = np.array([ys[i], xs[i]], float)
        if len(pts) and np.min(np.hypot(*(pts - p).T)) < mind:
            continue
        out.append(p)
        pts = np.vstack([pts, p])
    return np.array(out)


def make_qptiff(out: Path, seed=0):
    rng = np.random.default_rng(seed)
    H, W = 1500, 2600
    yy, xx = np.mgrid[:H, :W]
    pieces = [((750, 650), (560, 480), "Biopsy A", 1.0), ((750, 1900), (520, 560), "Biopsy B", 1.6)]
    imgs = {c: np.zeros((H, W), np.uint16) for c in CHANNELS}
    feats = []
    for (cy, cx), (ry, rx), name, gain in pieces:
        m = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
        cs = centers_in(m, 2600, 15, rng)
        im, *_ = paint((H, W), cs, rng, gain, m)
        for c in CHANNELS:
            imgs[c][m] = im[c][m]
        t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        ring = [[float(cx + (rx + 8) * np.cos(a)), float(cy + (ry + 8) * np.sin(a))] for a in t]
        ring.append(ring[0])
        feats.append({"type": "Feature", "id": name, "geometry": {"type": "Polygon", "coordinates": [ring]},
                      "properties": {"objectType": "annotation", "name": name.replace(" ", "_")}})
    for c in CHANNELS:  # background between pieces
        bg = imgs[c] == 0
        imgs[c][bg] = rng.poisson(60, bg.sum())
    res = (1e4 / 0.4977, 1)  # px per cm  -> 0.4977 µm/px
    with tifffile.TiffWriter(out) as tw:
        for c in CHANNELS:
            desc = (f'<?xml version="1.0" encoding="utf-16"?><PerkinElmer-QPI-ImageDescription><DescriptionVersion>2</DescriptionVersion>'
                    f'<ImageType>FullResolution</ImageType><Name>{c}</Name><Biomarker>{c}</Biomarker></PerkinElmer-QPI-ImageDescription>')
            tw.write(imgs[c], tile=(512, 512), description=desc, software="PerkinElmer-QPI", metadata=None, resolution=res, resolutionunit="CENTIMETER", compression="zlib")
        thumb = (imgs["DAPI"][::16, ::16] // 256).astype(np.uint8)
        tw.write(np.stack([thumb] * 3, -1), description='<PerkinElmer-QPI-ImageDescription><ImageType>Thumbnail</ImageType></PerkinElmer-QPI-ImageDescription>', software="PerkinElmer-QPI", metadata=None)
        for c in CHANNELS:
            tw.write(imgs[c][::2, ::2], tile=(512, 512), description=f"<PerkinElmer-QPI-ImageDescription><ImageType>ReducedResolution</ImageType><Name>{c}</Name></PerkinElmer-QPI-ImageDescription>",
                     software="PerkinElmer-QPI", metadata=None, compression="zlib")
    (out.parent / "outlines.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    (out.parent / "samples.csv").write_text("sample,batch,condition\nBiopsy_A,run1,control\nBiopsy_B,run2,disease\n")


def make_png(root: Path, seed=1):
    rng = np.random.default_rng(seed)
    for f, gain in (("fov1", 1.0), ("fov2", 1.3)):
        H = W = 500
        m = np.ones((H, W), bool)
        cs = centers_in(m, 900, 15, rng)
        imgs, *_ = paint((H, W), cs, rng, gain, None)
        d = root / f
        d.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(CHANNELS):
            Image.fromarray(imgs[c]).save(d / f"{i:02d}_{c}.png")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
    out.mkdir(parents=True, exist_ok=True)
    make_qptiff(out / "slide.qptiff")
    make_png(out / "png")
    print("written", out)
