"""Viewer feature table and gate evaluation.

`viewer_features` builds the exact values the HTML viewer plots (float32, 4 significant
digits); `evaluate_gates` applies a gates JSON saved from the viewer with the same
arithmetic as the viewer's JavaScript, so populations agree cell by cell."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .embed import cofactor, marker_names

log = logging.getLogger("vollmilch")

CAT_BASE = ["sample", "image"]
NON_FEATURE = {"cell_id", "object_id", "sample", "image", "has_nucleus", "x_px", "y_px"}


def round_sig(x, sig=4) -> np.ndarray:
    x = np.asarray(x, np.float64)
    out = x.copy()
    nz = np.isfinite(x) & (x != 0)
    if nz.any():
        mag = np.floor(np.log10(np.abs(x[nz])))
        f = 10.0 ** (sig - 1 - mag)
        out[nz] = np.round(x[nz] * f) / f
    return out.astype(np.float32)


def load_embeddings(cfg) -> dict[str, pd.DataFrame]:
    d = cfg.out / "embeddings"
    out = {}
    if d.exists():
        for p in sorted(d.glob("*.parquet")):
            out[p.stem] = pd.read_parquet(p)
    return out


def viewer_features(df: pd.DataFrame, cfg, embeddings: dict | None = None):
    """Returns (numeric DataFrame of float32, list of column descriptors, categorical dict)."""
    rc = cfg["report"]
    cols, desc = {}, []
    markers = marker_names(df)
    for comp in rc.get("compartments") or ["cell", "nucleus"]:
        for st in rc.get("stats") or ["mean"]:
            for m in markers:
                c = f"{m}:{comp}:{st}"
                if c not in df.columns:
                    continue
                cf = cofactor(cfg, m)
                cols[c] = round_sig(np.arcsinh(df[c].to_numpy(np.float64) / cf))
                desc.append({"key": c, "label": f"{m}", "group": f"Markers: {comp} {st} (asinh)", "marker": m, "cofactor": cf})
    for c in df.columns:
        if c.startswith(("nucleus:", "cell:")) and c.count(":") == 1:
            cols[c] = round_sig(df[c].to_numpy(np.float64))
            grp = "Nucleus" if c.startswith("nucleus:") else "Cell shape"
            desc.append({"key": c, "label": c.split(":", 1)[1], "group": grp})
    for c in ("x_um", "y_um"):
        cols[c] = round_sig(df[c].to_numpy(np.float64))
        desc.append({"key": c, "label": c, "group": "Position"})
    cats = {}
    for c in CAT_BASE:
        cats[c] = df[c].astype(str).to_numpy()
    st_cols = [c for c in df.columns if c not in cols and c not in NON_FEATURE and not c.startswith(("nucleus:", "cell:"))
               and c.count(":") == 0 and c not in ("x_um", "y_um") and c not in CAT_BASE
               and (df[c].dtype == object or pd.api.types.is_string_dtype(df[c]))]
    for c in st_cols:
        cats[c] = df[c].fillna("NA").astype(str).to_numpy()
    for name, e in (embeddings or {}).items():
        e = e.set_index("cell_id").reindex(df["cell_id"])
        for v, lab in (("", "uncorrected"), ("_harmony", "Harmony")):
            if f"umap{v}_1" not in e.columns:
                continue
            for k in (1, 2):
                key = f"emb:{name}:umap{v}_{k}"
                cols[key] = round_sig(e[f"umap{v}_{k}"].to_numpy(np.float64))
                desc.append({"key": key, "label": f"UMAP {k} ({name}, {lab})", "group": "Embedding", "embedding": name, "variant": lab, "dim": k})
            lc = e[f"leiden{v}"]
            cats[f"leiden {name}{'' if not v else ' harmony'}"] = np.where(lc.isna(), "NA", lc.fillna(-1).astype(int).astype(str))
    if markers and not any(d.get("marker") for d in desc):
        log.warning("no marker columns for report.stats %s in the cell table; if the samples were segmented with other "
                    "features.stats, run `vollmilch segment --reuse-segmentation`", rc.get("stats"))
    return pd.DataFrame(cols), desc, cats


# ---------------------------------------------------------------------------- gates
def load_gates(path) -> dict:
    g = json.loads(Path(path).read_text(encoding="utf-8"))
    if g.get("format") not in ("vollmilch-gates", "cellgate-gates"):  # cellgate = former name
        raise ValueError(f"{path} is not a vollmilch gates file")
    return g


def _axis(values: pd.DataFrame, key, log10):
    if key not in values.columns:
        raise KeyError(f"gate feature {key!r} not available (embedding missing or different config?)")
    v = values[key].to_numpy(np.float32).astype(np.float64)
    if log10:
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(v > 0, np.log10(np.where(v > 0, v, 1.0)), np.nan)
    return v


def in_polygon(px, py, verts) -> np.ndarray:
    """Even-odd ray casting; same expression order as the viewer."""
    inside = np.zeros(px.shape, bool)
    n = len(verts)
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            cond = ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / (yj - yi) + xi)
        inside ^= cond
        j = i
    return inside & np.isfinite(px) & np.isfinite(py)


def gate_mask(values: pd.DataFrame, gate: dict) -> np.ndarray:
    x = _axis(values, gate["x"], gate.get("logx", False))
    y = _axis(values, gate["y"], gate.get("logy", False))
    if gate["type"] == "rect":
        m = np.isfinite(x) & np.isfinite(y)
        if gate.get("x0") is not None:
            m &= x >= gate["x0"]
        if gate.get("x1") is not None:
            m &= x <= gate["x1"]
        if gate.get("y0") is not None:
            m &= y >= gate["y0"]
        if gate.get("y1") is not None:
            m &= y <= gate["y1"]
        return m
    return in_polygon(x, y, gate["vertices"])


def tree_order(pops: list[dict]) -> list[dict]:
    kids: dict = {}
    for p in pops:
        kids.setdefault(p.get("parent"), []).append(p)
    out = []

    def walk(pid, depth):
        for p in kids.get(pid, []):
            out.append({**p, "_depth": depth})
            walk(p["id"], depth + 1)

    walk(None, 0)
    return out


def evaluate_gates(values: pd.DataFrame, gates: dict):
    """Returns (membership dict id->bool array, ordered populations, deepest population index per cell (-1 = none))."""
    pops = tree_order(gates.get("populations", []))
    member = {}
    n = len(values)
    for p in pops:
        parent = member[p["parent"]] if p.get("parent") else np.ones(n, bool)
        member[p["id"]] = parent & gate_mask(values, p)
    best = np.full(n, -1)
    best_depth = np.full(n, -1)
    for i, p in enumerate(pops):
        m = member[p["id"]] & (p["_depth"] > best_depth)
        best[m] = i
        best_depth[m] = p["_depth"]
    return member, pops, best


def population_path(pops, i, sep="/"):
    byid = {p["id"]: p for p in pops}
    p = pops[i]
    parts = [p["name"]]
    while p.get("parent"):
        p = byid[p["parent"]]
        parts.append(p["name"])
    return sep.join(reversed(parts))
