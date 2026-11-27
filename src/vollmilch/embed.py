"""PCA / UMAP / Leiden, uncorrected and Harmony-corrected, on all cells or a gated population."""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

log = logging.getLogger("vollmilch")

MORPH_FEATURES = ["nucleus:area_um2", "nucleus:circularity", "nucleus:solidity", "nucleus:eccentricity",
                  "nucleus:dapi_cv", "cell:area_um2", "cell:circularity", "cell:solidity", "cell:nc_ratio"]
LOG_FEATURES = {"nucleus:area_um2", "cell:area_um2"}


def cofactor(cfg, marker: str) -> float:
    return float((cfg["transform"].get("cofactors") or {}).get(marker, cfg["transform"].get("cofactor", 5)))


def marker_names(df: pd.DataFrame) -> list[str]:
    seen = []
    for c in df.columns:
        p = c.split(":")
        if len(p) == 3 and p[1] in ("nucleus", "cell", "cytoplasm") and p[0] not in seen:
            seen.append(p[0])
    return seen


def embedding_markers(df, cfg, markers=None):
    ec = cfg["embedding"]
    comp, stat = ec.get("compartment", "cell"), ec.get("stat", "mean")
    if f"{marker_names(df)[0]}:{comp}:{stat}" not in df.columns and comp == "cell":
        comp = "nucleus"  # nuclei mode has no separate cell compartment
    ms = markers or (marker_names(df) if ec["markers"] in ("all", None) else list(ec["markers"]))
    ms = [m for m in ms if m not in set(ec.get("exclude") or [])]
    cols = [f"{m}:{comp}:{stat}" for m in ms]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"embedding features not in the cell table: {miss}")
    return ms, cols


def feature_matrix(df, cfg, markers=None):
    ms, cols = embedding_markers(df, cfg, markers)
    X = np.column_stack([np.arcsinh(df[c].to_numpy(np.float64) / cofactor(cfg, m)) for m, c in zip(ms, cols)])
    names = list(ms)
    if cfg["embedding"].get("include_morphology"):
        for f in MORPH_FEATURES:
            if f in df.columns:
                v = df[f].to_numpy(np.float64)
                if f in LOG_FEATURES:
                    v = np.log(np.clip(v, 1e-3, None))
                X = np.column_stack([X, v])
                names.append(f)
    # NaN (e.g. cells without nucleus) -> column median
    med = np.nanmedian(X, axis=0)
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.take(np.nan_to_num(med), np.nonzero(bad)[1])
    return X.astype(np.float32), names


def _umap_leiden(ad, rep, key, cfg):
    import scanpy as sc

    ec = cfg["embedding"]
    seed = ec.get("seed", 0)
    t = time.time()
    sc.pp.neighbors(ad, use_rep=rep, n_neighbors=ec.get("n_neighbors", 15), random_state=seed, key_added=key)
    sc.tl.umap(ad, neighbors_key=key, random_state=seed)
    umap = ad.obsm["X_umap"].copy()
    kw = dict(resolution=ec.get("leiden_resolution", 1.0), neighbors_key=key, key_added=f"leiden_{key}", random_state=seed)
    try:
        sc.tl.leiden(ad, flavor="igraph", n_iterations=2, directed=False, **kw)
    except TypeError:  # older scanpy
        sc.tl.leiden(ad, **kw)
    log.info("  %s: neighbours, UMAP and Leiden in %.0f s (%d clusters)", key, time.time() - t,
             ad.obs[f"leiden_{key}"].nunique())
    return umap, ad.obs[f"leiden_{key}"].astype(int).to_numpy()


def run_embedding(cfg, df: pd.DataFrame, name: str = "all", markers=None, batch_key="config"):
    try:
        import anndata as ad_mod
        import scanpy as sc
    except ImportError as e:
        raise ImportError("embedding needs scanpy: pip install 'vollmilch[embed]'") from e
    from sklearn.neighbors import KNeighborsClassifier

    ec = cfg["embedding"]
    seed = ec.get("seed", 0)
    bk = ec.get("batch_key") if batch_key == "config" else batch_key
    X, feat_names = feature_matrix(df, cfg, markers)
    n = len(df)
    log.info("embedding '%s': %d cells x %d features", name, n, X.shape[1])
    rng = np.random.default_rng(seed)
    m = int(ec.get("max_cells", 200000))
    sub = np.sort(rng.choice(n, m, replace=False)) if n > m else np.arange(n)
    ad = ad_mod.AnnData(X[sub].copy())
    ad.var_names = [f.replace(":", "_") for f in feat_names]
    sc.pp.scale(ad, max_value=10)
    k = int(min(ec.get("n_pcs", 20), X.shape[1] - 1, len(sub) - 1))
    sc.pp.pca(ad, n_comps=k, random_state=seed)
    mean, std = ad.var["mean"].to_numpy().copy(), ad.var["std"].to_numpy().copy()
    std[std == 0] = 1

    res = pd.DataFrame({"cell_id": df["cell_id"].to_numpy()})
    in_emb = np.zeros(n, bool)
    in_emb[sub] = True
    res["in_embedding"] = in_emb

    def full_pcs():
        Z = np.clip((X - mean) / std, -10, 10)
        Z = Z - Z[sub].mean(0)
        return Z @ ad.varm["PCs"]

    pcs_all = full_pcs() if len(sub) < n else ad.obsm["X_pca"]

    variants = [("pca", "X_pca")]
    if bk:
        if bk not in df.columns:
            raise KeyError(f"batch key {bk!r} not in the cell table (columns from samples_table are allowed)")
        batches = df[bk].astype(str).to_numpy()[sub]
        if len(set(batches)) < 2:
            log.warning("  only one batch in %r; Harmony skipped", bk)
        else:
            import harmonypy

            t = time.time()
            ho = harmonypy.run_harmony(ad.obsm["X_pca"], pd.DataFrame({"batch": batches}), ["batch"],
                                       random_state=seed, verbose=False)
            Z = np.asarray(ho.Z_corr)
            if Z.shape[0] != len(sub):
                Z = Z.T
            ad.obsm["X_harmony"] = Z
            variants.append(("harmony", "X_harmony"))
            log.info("  Harmony on %r (%d batches) in %.0f s", bk, len(set(batches)), time.time() - t)

    for key, rep in variants:
        umap, leiden = _umap_leiden(ad, rep, key, cfg)
        u = np.full((n, 2), np.nan, np.float32)
        u[sub] = umap
        lab = np.zeros(n, int)
        lab[sub] = leiden
        if len(sub) < n:
            clf = KNeighborsClassifier(n_neighbors=15).fit(pcs_all[sub], leiden)
            rest = np.flatnonzero(~in_emb)
            for i in range(0, len(rest), 200000):
                part = rest[i:i + 200000]
                lab[part] = clf.predict(pcs_all[part])
        sfx = "" if key == "pca" else "_harmony"
        res[f"umap{sfx}_1"] = u[:, 0]
        res[f"umap{sfx}_2"] = u[:, 1]
        res[f"leiden{sfx}"] = lab

    outd = cfg.out / "embeddings"
    outd.mkdir(parents=True, exist_ok=True)
    res.to_parquet(outd / f"{name}.parquet", index=False)
    meta = {"name": name, "n_cells": n, "n_embedded": int(len(sub)), "features": feat_names,
            "batch_key": bk if len(variants) > 1 else None, "n_pcs": k,
            "params": {kk: ec.get(kk) for kk in ("n_neighbors", "leiden_resolution", "seed", "compartment", "stat")}}
    (outd / f"{name}.json").write_text(json.dumps(meta, indent=1))
    log.info("embedding '%s' written", name)
    return res
