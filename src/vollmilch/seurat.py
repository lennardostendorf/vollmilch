"""Export to Seurat: a 10x-style matrix folder (Read10X reads it), cell metadata, UMAPs and the
x/y position, plus make_seurat.R, which builds the Seurat object. The script is run when R with
Seurat is installed, giving seurat.rds; otherwise it can be run later, or the files read by hand.

    seurat/
      matrix.mtx.gz features.tsv.gz barcodes.tsv.gz   raw intensities, markers x cells (counts)
      asinh.mtx.gz                                    asinh(intensity / cofactor) (data)
      metadata.csv.gz                                 sample, position, morphology, Leiden, populations
      reduction_<name>.csv.gz                         umap_<embedding>[_harmony], spatial (x, -y in um)
      make_seurat.R  seurat.rds
"""
from __future__ import annotations

import gzip
import logging
import shutil
import subprocess
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io, sparse

from .embed import cofactor, marker_names
from .gating import evaluate_gates, load_embeddings, load_gates, population_path, viewer_features
from .pipeline import load_cells

log = logging.getLogger("vollmilch")

META_SKIP = {"cell_id"}


def _r_name(c: str) -> str:
    """Column names R users can type: nucleus:area_um2 -> nucleus_area_um2."""
    return c.replace(":", "_").replace(" ", "_")


def seurat_available() -> bool:
    if not shutil.which("Rscript"):
        return False
    r = subprocess.run(["Rscript", "-e", 'quit(status = !requireNamespace("Seurat", quietly = TRUE))'],
                       capture_output=True)
    return r.returncode == 0


def to_seurat(cfg, out_dir=None, gates_path=None, compartment=None, stat=None, assay="Protein",
              samples=None, rds=True) -> Path:
    df = load_cells(cfg, samples)
    ec = cfg["embedding"]
    comp, stat = compartment or ec.get("compartment", "cell"), stat or ec.get("stat", "mean")
    markers = marker_names(df)
    if markers and f"{markers[0]}:{comp}:{stat}" not in df.columns and comp == "cell":
        comp = "nucleus"  # nuclei mode has no separate cell compartment
    cols = [f"{m}:{comp}:{stat}" for m in markers]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"not in the cell table: {miss}; available statistics come from features.stats")
    out = Path(out_dir) if out_dir else cfg.out / "seurat"
    out.mkdir(parents=True, exist_ok=True)
    cells = df["cell_id"].astype(str).to_numpy()

    # --- matrices, markers x cells; zeros are not stored (background-subtracted data is sparse)
    X = df[cols].to_numpy(np.float64)
    n_nan = int(np.isnan(X).sum())
    if n_nan:
        log.warning("%d missing values (e.g. %s of cells without nucleus) written as 0", n_nan, comp)
    X = np.nan_to_num(X)
    cf = np.array([cofactor(cfg, m) for m in markers])
    feats = [m.replace("_", "-") for m in markers]  # Seurat replaces underscores in feature names anyway
    for name, M in (("matrix.mtx.gz", X), ("asinh.mtx.gz", np.arcsinh(X / cf))):
        with gzip.open(out / name, "wb") as fh:
            io.mmwrite(fh, sparse.coo_matrix(M.T), precision=6)
    with gzip.open(out / "features.tsv.gz", "wt") as fh:
        fh.writelines(f"{f}\t{m}\tAntibody Capture\n" for f, m in zip(feats, markers))
    with gzip.open(out / "barcodes.tsv.gz", "wt") as fh:
        fh.writelines(f"{c}\n" for c in cells)

    # --- metadata: everything per cell that is not a marker intensity
    meta = df[[c for c in df.columns if c not in META_SKIP and not (c.count(":") == 2 and c.split(":")[0] in markers)]].copy()
    meta.columns = [_r_name(c) for c in meta.columns]
    embs = load_embeddings(cfg)
    for name, e in embs.items():
        e = e.set_index("cell_id").reindex(df["cell_id"])
        for v in ("", "_harmony"):
            if f"umap{v}_1" not in e.columns:
                continue
            lc = e[f"leiden{v}"]
            meta[f"leiden_{name}{v}"] = np.where(lc.isna(), "NA", lc.fillna(-1).astype(int).astype(str))
            pd.DataFrame({"cell": cells, "1": e[f"umap{v}_1"].to_numpy(), "2": e[f"umap{v}_2"].to_numpy()}).to_csv(
                out / f"reduction_umap_{name}{v}.csv.gz", index=False)
    # y negated: image rows count downwards, plots count upwards; so plots match the image and QuPath
    pd.DataFrame({"cell": cells, "1": df["x_um"].to_numpy(), "2": -df["y_um"].to_numpy()}).to_csv(
        out / "reduction_spatial.csv.gz", index=False)
    if gates_path:
        vals, _, _ = viewer_features(df, cfg, embs)
        member, pops, best = evaluate_gates(vals, load_gates(gates_path))
        names = np.array(["none"] + [p["name"] for p in pops], dtype=object)
        paths = np.array(["none"] + [population_path(pops, i) for i in range(len(pops))], dtype=object)
        meta["population"], meta["population_path"] = names[best + 1], paths[best + 1]
        for p in pops:
            meta[_r_name(f"gate_{p['name']}")] = member[p["id"]]
    meta.insert(0, "cell", cells)
    meta.to_csv(out / "metadata.csv.gz", index=False)

    script = out / "make_seurat.R"
    script.write_text(resources.files("vollmilch").joinpath("make_seurat.R").read_text(encoding="utf-8")
                      .replace("__ASSAY__", assay), encoding="utf-8")
    log.info("Seurat files for %d cells x %d markers (%s %s) in %s", len(cells), len(markers), comp, stat, out)
    if rds:
        if seurat_available():
            log.info("building seurat.rds with R ...")
            r = subprocess.run(["Rscript", str(script), str(out)], capture_output=True, text=True)
            if r.returncode:
                log.error("make_seurat.R failed:\n%s", r.stderr[-3000:])
                raise RuntimeError("make_seurat.R failed; the matrix files are complete, see the R error above")
            log.info("%s", r.stdout.strip())
        else:
            log.warning("R with Seurat not found: run `Rscript %s` where it is installed (or Read10X(\"%s\") by hand)",
                        script, out)
    return out
