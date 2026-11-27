"""Apply a gates JSON to all cells and write QuPath GeoJSON + class tables."""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .embed import marker_names
from .gating import evaluate_gates, load_embeddings, load_gates, population_path, viewer_features
from .pipeline import load_cells, read_labels, sample_meta

log = logging.getLogger("vollmilch")


def _rgb(hexcol: str):
    h = hexcol.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def _contours(lab: np.ndarray, offset_xy):
    """dict object id -> list of [x, y] (closed ring, image pixel coordinates)."""
    out = {}
    ox, oy = offset_xy
    for oid, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        sub = (lab[sl] == oid).astype(np.uint8)
        cs, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float64)
        x0, y0 = sl[1].start + ox, sl[0].start + oy
        if len(c) < 3:  # 1–2 px wide objects: use the bounding box
            h, w = sub.shape
            ring = [[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]]
        else:
            ring = (c + (x0 + 0.5, y0 + 0.5)).round(2).tolist()
        ring.append(ring[0])
        out[oid] = ring
    return out


def export_gates(cfg, gates_path, samples=None, only=None, measurements="markers", full_path=False,
                 out_dir=None, geometry=True):
    gates = load_gates(gates_path)
    df = load_cells(cfg, samples)
    embs = load_embeddings(cfg)
    vals, desc, _ = viewer_features(df, cfg, embs)
    member, pops, best = evaluate_gates(vals, gates)
    out_dir = Path(out_dir) if out_dir else cfg.out / "export" / Path(gates_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    sep = ": " if full_path else "/"
    names = np.array([""] + [p["name"] for p in pops], dtype=object)
    paths = np.array([""] + [population_path(pops, i) for i in range(len(pops))], dtype=object)
    qnames = np.array([""] + [population_path(pops, i, ": ") if full_path else p["name"] for i, p in enumerate(pops)], dtype=object)
    colors = [None] + [_rgb(p.get("color", "#888888")) for p in pops]

    cls = pd.DataFrame({"cell_id": df["cell_id"], "sample": df["sample"], "image": df["image"], "object_id": df["object_id"],
                        "x_px": df["x_px"].round(2), "y_px": df["y_px"].round(2),
                        "population": names[best + 1], "path": paths[best + 1]})
    for p in pops:
        cls[f"gate:{p['name']}"] = member[p["id"]].astype(np.uint8)
    cls.to_csv(out_dir / "classes.csv", index=False)

    rows = []
    for s, idx in cls.groupby("sample").indices.items():
        tot = len(idx)
        for p in pops:
            n = int(member[p["id"]][idx].sum())
            par = int(member[p["parent"]][idx].sum()) if p.get("parent") else tot
            rows.append({"sample": s, "population": p["name"], "path": population_path(pops, pops.index(p)),
                         "count": n, "parent_count": par, "pct_parent": round(100 * n / par, 3) if par else np.nan,
                         "sample_total": tot, "pct_total": round(100 * n / tot, 3) if tot else np.nan})
    pd.DataFrame(rows).to_csv(out_dir / "population_counts.csv", index=False)
    log.info("classes for %d cells written to %s", len(cls), out_dir)
    if not geometry:
        return out_dir

    keep = np.ones(len(df), bool)
    if only:
        sel = [p for p in pops if p["name"] == only]
        if not sel:
            raise KeyError(f"population {only!r} not in {gates_path}")
        keep = member[sel[0]["id"]]
    # measurements
    meas_cols = []
    if measurements in ("markers", "all"):
        for m in marker_names(df):
            for c in df.columns:
                if c.startswith(f"{m}:") and c.count(":") == 2:
                    meas_cols.append((c, f"{m}: {c.split(':')[1]} {c.split(':')[2]}"))
    if measurements == "all":
        meas_cols += [(c, c.replace(":", ": ")) for c in df.columns if c.startswith(("nucleus:", "cell:")) and c.count(":") == 1]
    gate_cols = [(f"gate:{p['name']}", f"Gate: {p['name']}") for p in pops]

    mode = cfg["segmentation"]["mode"]
    for image, gidx in df.groupby("image").indices.items():
        path = out_dir / f"{image}_cells.geojson"
        n_written = 0
        with open(path, "w", encoding="utf-8") as fh:
            fh.write('{"type":"FeatureCollection","features":[\n')
            first = True
            for s in df["sample"].iloc[gidx].unique():
                meta = sample_meta(cfg, s)
                y0, _, x0, _ = meta["bbox"]
                sd = cfg.sample_dir(s)
                cell_lab = read_labels(sd / "cell_labels.tif")
                cell_c = _contours(cell_lab, (x0, y0))
                del cell_lab
                nuc_c = {}
                if mode != "nuclei":
                    nuc_lab = read_labels(sd / "nuclei_labels.tif")
                    nuc_c = _contours(nuc_lab, (x0, y0))
                    del nuc_lab
                sidx = np.flatnonzero((df["sample"].to_numpy() == s) & keep)
                oids = df["object_id"].to_numpy()[sidx]
                meas = {q: df[c].to_numpy()[sidx] for c, q in meas_cols}
                gmeas = {q: cls[c].to_numpy()[sidx] for c, q in gate_cols}
                bsel = best[sidx]
                for k, oid in enumerate(oids):
                    ring = cell_c.get(int(oid))
                    if ring is None:
                        continue
                    cid = f"{s}:{oid}"
                    props = {"objectType": "cell" if mode != "nuclei" else "detection", "name": cid}
                    b = bsel[k]
                    if b >= 0:
                        props["classification"] = {"name": qnames[b + 1], "color": colors[b + 1]}
                    m = {q: (None if not np.isfinite(v[k]) else round(float(v[k]), 4)) for q, v in meas.items()}
                    m.update({q: int(v[k]) for q, v in gmeas.items()})
                    props["measurements"] = {q: v for q, v in m.items() if v is not None}
                    feat = {"type": "Feature", "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "cellgate:" + cid)),  # seed from the former name keeps object IDs stable
                            "geometry": {"type": "Polygon", "coordinates": [ring]}}
                    nr = nuc_c.get(int(oid))
                    if nr is not None:
                        feat["nucleusGeometry"] = {"type": "Polygon", "coordinates": [nr]}
                    feat["properties"] = props
                    fh.write(("" if first else ",\n") + json.dumps(feat, separators=(",", ":")))
                    first = False
                    n_written += 1
            fh.write("\n]}\n")
        log.info("%s: %d objects -> %s", image, n_written, path)
    return out_dir
