"""Cofactor suggestion, the cofactor CSV in the config, and the viewer build:  pytest tests/"""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import make_synthetic as ms  # noqa: E402

from vollmilch.cofactors import p1_positive, run_cofactors, sig2, suggest  # noqa: E402
from vollmilch.config import load_config, read_cofactor_table  # noqa: E402
from vollmilch.embed import cofactor  # noqa: E402

N = 400_000


def _mix(rng, bg, pos, frac_pos):
    k = int(N * frac_pos)
    return np.r_[bg(N - k), pos(k)]


@pytest.mark.parametrize("name, make, lo, hi", [
    # background mode + 3 SD: 150 + 30
    ("raw fluorescence", lambda r: _mix(r, lambda n: r.normal(150, 10, n), lambda n: r.lognormal(np.log(2000), .5, n), .1).round(), 170, 190),
    ("mostly positive", lambda r: _mix(r, lambda n: r.normal(150, 10, n), lambda n: r.lognormal(np.log(2000), .5, n), .6).round(), 170, 190),
    # background subtracted, around 0: 3 x 20
    ("background subtracted", lambda r: _mix(r, lambda n: r.normal(0, 20, n), lambda n: r.lognormal(np.log(800), .6, n), .2), 55, 65),
    ("subtracted, clipped at 0", lambda r: np.clip(_mix(r, lambda n: r.normal(0, 20, n), lambda n: r.lognormal(np.log(800), .6, n), .2), 0, None), 50, 65),
    ("8-bit", lambda r: np.clip(_mix(r, lambda n: r.normal(12, 3, n), lambda n: r.normal(120, 30, n), .15), 0, 255).round(), 19, 23),
    ("scaled to 0-1", lambda r: _mix(r, lambda n: r.normal(.02, .004, n), lambda n: r.normal(.4, .1, n), .1), .029, .035),
    # conventions: CyTOF about 5, IMC counts about 1-5
    ("CyTOF-like", lambda r: _mix(r, lambda n: np.clip(r.normal(0, 1.5, n), 0, None), lambda n: r.lognormal(np.log(80), .8, n), .5), 3.5, 5.5),
    ("IMC-like counts", lambda r: _mix(r, lambda n: r.poisson(.5, n), lambda n: r.poisson(30, n), .2).astype(float), 1, 5),
])
def test_suggest_known_distributions(name, make, lo, hi):
    s = suggest(make(np.random.default_rng(0)))
    assert lo <= s["cofactor"] <= hi, (name, s)
    assert s["frac_below"] < 0.05, (name, s)


def test_suggest_few_cells():
    """A few hundred cell means: the background SD must still be resolved."""
    sds = []
    for seed in range(10):
        r = np.random.default_rng(seed)
        v = np.r_[r.normal(100, 1, 240), r.lognormal(np.log(700), .6, 60)]
        sds.append(suggest(v)["noise_sd"])
    assert 0.7 < np.median(sds) < 1.3, sds


def test_suggest_flags_ubiquitous_marker():
    r = np.random.default_rng(1)
    # a narrow background is the tallest peak even as a minority: suggestion still right
    s = suggest(np.r_[r.normal(100, 5, 30_000), r.normal(600, 60, 70_000)])
    assert 110 <= s["cofactor"] <= 120 and s["frac_below"] < 0.05
    # positives form the tallest peak: flagged
    s = suggest(np.r_[r.normal(100, 20, 20_000), r.normal(400, 15, 80_000)])
    assert s["frac_below"] > 0.15


def test_background_removed():
    """Thresholded images (most pixels exactly 0, signal above) have no background to measure."""
    r = np.random.default_rng(2)
    signal = r.lognormal(np.log(5000), 1, 20_000)
    px = np.r_[np.zeros(180_000), signal].round()
    s = suggest(px, "pixels")
    assert s["background_removed"] and s["method"] == "none" and np.isnan(s["cofactor"]) and abs(s["zero_frac"] - 0.9) < 1e-9
    cells = np.r_[np.zeros(30), r.lognormal(np.log(200), 2, 2970)]  # 1 % of cells all 0
    assert suggest(cells, "cells")["background_removed"]
    # not removed: Poisson counts (0 is only ~2x as common as 1) and clipped noisy background
    assert not suggest(np.r_[r.poisson(.5, 160_000), r.poisson(30, 40_000)].astype(float), "pixels")["background_removed"]
    clipped = np.clip(np.r_[r.normal(0, 20, 160_000), r.lognormal(np.log(800), .6, 40_000)], 0, None)
    assert not suggest(clipped, "pixels")["background_removed"]
    # saturation does not become the background peak
    sat = np.r_[r.normal(150, 10, 180_000), np.full(20_000, 65535.0)].round()
    assert 170 <= suggest(sat)["cofactor"] <= 190


def test_p1_positive():
    v = np.r_[np.zeros(500), np.arange(1, 101.0)]
    assert p1_positive(v) == np.percentile(np.arange(1, 101.0), 1)
    assert np.isnan(p1_positive(np.r_[np.zeros(100), 5.0]))  # too few positive values


def test_suggest_degenerate():
    assert np.isnan(suggest(np.arange(10.0))["cofactor"])
    assert suggest(np.full(1000, 7.0))["cofactor"] > 0


def test_cofactor_table(tmp_path):
    p = tmp_path / "cf.csv"
    p.write_text("Marker,Cofactor,suggested\nCD3,20,18\nCD68,150,150\nKi67,,5\n")
    assert read_cofactor_table(p) == {"CD3": 20.0, "CD68": 150.0}
    p.write_text("marker\tcofactor\nCD3\t20\n")  # tab separated
    assert read_cofactor_table(p) == {"CD3": 20.0}
    for bad in ("marker,cofactor\nCD3,0\n", "marker,cofactor\nCD3,x\n", "marker,cofactor\nCD3,1\nCD3,2\n", "name,value\nCD3,1\n"):
        p.write_text(bad)
        with pytest.raises(ValueError):
            read_cofactor_table(p)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    ms.make_qptiff(d / "slide.qptiff")
    return d


def _config(d: Path, extra="", out="out"):
    p = d / f"config_{out}.yaml"
    p.write_text(f"""project: test
output: {out}
source: {{type: qptiff, path: slide.qptiff}}
regions: {{type: geojson, path: outlines.geojson}}
segmentation: {{backend: watershed, mode: expand}}
{extra}""")
    return p


def test_config_reads_cofactor_csv(synthetic):
    (synthetic / "cf.csv").write_text("marker,cofactor\nCD3,20\n")
    cfg = load_config(_config(synthetic, "transform: {cofactor: 5, cofactors: cf.csv}\n"))
    assert cofactor(cfg, "CD3") == 20 and cofactor(cfg, "CD68") == 5
    assert Path(cfg["transform"]["_cofactors_file"]) == (synthetic / "cf.csv").resolve()


def test_viewer_from_pixels(synthetic):
    shutil.rmtree(synthetic / "out", ignore_errors=True)
    cfg = load_config(_config(synthetic))
    html = run_cofactors(cfg, level="pixels", max_pixels=200_000).read_text()
    m = json.loads(re.search(r'<script id="manifest" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert m["samples"] == ["Biopsy_A", "Biopsy_B"]
    assert [x["name"] for x in m["markers"]] == ms.CHANNELS
    for x in m["markers"]:
        # synthetic background: Poisson(80 + U(0, 40)), so mode ~100 and SD ~15
        assert 120 <= x["suggested"] <= 180, x
        assert x["current"] == 5 and x["slider"][0] < 5 < x["suggested"] < x["slider"][1]
        assert len(m["blob"][f"hist:{m['markers'].index(x)}"]) and m["blob"][f"hist:{m['markers'].index(x)}"]["length"] == 2 * m["n_fine"]
    assert "__" not in re.sub(r"<script.*?</script>", "", html, flags=re.S)  # every placeholder filled
    # suggested.csv is itself a valid cofactor table
    t = read_cofactor_table(cfg.out / "cofactors" / "suggested.csv")
    assert set(t) == set(ms.CHANNELS)


def _manifest(html: str) -> dict:
    return json.loads(re.search(r'<script id="manifest" type="application/json">(.*?)</script>', html, re.S).group(1))


def test_auto_uses_cells_once_segmented(synthetic):
    from vollmilch.pipeline import run_segmentation

    cfg = load_config(_config(synthetic, out="out_auto"))
    assert _manifest(run_cofactors(cfg, max_pixels=100_000).read_text())["level"] == "pixels"
    run_segmentation(cfg)
    m = _manifest(run_cofactors(cfg).read_text())
    assert m["level"] == "cells"
    for x in m["markers"][1:]:  # cell means: background ~100 with a small SD (DAPI: every cell is positive)
        assert 95 <= x["suggested"] <= 130 and {p["method"] for p in x["per_sample"]} == {"background"}, x


def test_to_seurat(synthetic):
    """Matrix files readable as 10x data; with R + Seurat installed also a loadable seurat.rds."""
    import gzip
    import subprocess

    import pandas as pd
    from scipy import io

    from vollmilch.pipeline import run_segmentation
    from vollmilch.seurat import seurat_available, to_seurat

    cfg = load_config(_config(synthetic, out="out_seurat"))
    run_segmentation(cfg)
    rds = seurat_available()
    out = to_seurat(cfg, rds=rds)
    with gzip.open(out / "matrix.mtx.gz") as fh:
        counts = io.mmread(fh).toarray()
    with gzip.open(out / "asinh.mtx.gz") as fh:
        data = io.mmread(fh).toarray()
    feats = pd.read_csv(out / "features.tsv.gz", sep="\t", header=None)[0].tolist()
    cells = pd.read_csv(out / "barcodes.tsv.gz", header=None)[0].tolist()
    meta = pd.read_csv(out / "metadata.csv.gz", index_col=0)
    assert feats == ms.CHANNELS and counts.shape == (len(feats), len(cells)) and list(meta.index) == cells
    df = pd.concat([pd.read_parquet(cfg.sample_dir(s) / "cells.parquet") for s in ("Biopsy_A", "Biopsy_B")])
    assert np.allclose(counts[feats.index("CD3")], df["CD3:cell:mean"], rtol=1e-5)
    assert np.allclose(data, np.arcsinh(counts / 5), rtol=1e-4, atol=1e-6)  # default cofactor 5
    assert "nucleus_area_um2" in meta.columns and not any(":" in c for c in meta.columns)
    if rds:
        r = subprocess.run(["Rscript", "-e", f'suppressMessages(library(SeuratObject)); o <- readRDS("{out / "seurat.rds"}"); cat(ncol(o), nrow(o), Reductions(o))'],
                           capture_output=True, text=True)
        assert r.stdout.split()[:3] == [str(len(cells)), str(len(feats)), "spatial"], r.stdout + r.stderr


def test_config_stats_follow_features(synthetic):
    """embedding.stat / report.stats not in features.stats fall back to features.stats, with a warning
    (a config from `vollmilch init` lists embedding.stat: mean even after features.stats is changed)."""
    cfg = load_config(_config(synthetic, "features: {stats: [median]}\n", out="out_stats1"))
    assert cfg["embedding"]["stat"] == "median" and cfg["report"]["stats"] == ["median"]
    cfg = load_config(_config(synthetic, "features: {stats: [median]}\nembedding: {stat: mean}\nreport: {stats: [mean]}\n",
                              out="out_stats2"))  # what an edited `vollmilch init` config looks like
    assert cfg["embedding"]["stat"] == "median" and cfg["report"]["stats"] == ["median"]
    cfg = load_config(_config(synthetic, "features: {stats: [mean, median]}\nreport: {stats: [median, p90]}\n", out="out_stats3"))
    assert cfg["embedding"]["stat"] == "mean" and cfg["report"]["stats"] == ["median"]


@pytest.fixture(scope="module")
def subtracted(synthetic):
    """The synthetic slide background-subtracted and clipped at 0, like MELC: most pixels exactly 0."""
    import tifffile

    with tifffile.TiffFile(synthetic / "slide.qptiff") as tf:
        a = np.stack([p.asarray() for p in tf.series[0].pages[:len(ms.CHANNELS)]]).astype(np.int32)
    a = np.clip(a - 200, 0, None).astype(np.uint16)  # background ~100 +- 15: all of it goes to 0
    tifffile.imwrite(synthetic / "sub.ome.tif", a, ome=True, photometric="minisblack",
                     metadata={"axes": "CYX", "Channel": {"Name": ms.CHANNELS}, "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5})
    p = synthetic / "config_sub.yaml"
    p.write_text("project: sub\noutput: out_sub\nsource: {type: ome_tiff, path: sub.ome.tif}\n"
                 "regions: {type: geojson, path: outlines.geojson}\nsegmentation: {backend: watershed, mode: expand}\n")
    return p


def test_subtracted_rules(subtracted):
    import pandas as pd

    from vollmilch.pipeline import run_segmentation

    cfg = load_config(subtracted)
    per_sample = lambda: pd.read_csv(cfg.out / "cofactors" / "per_sample.csv")  # noqa: E731
    # before segmentation: the default (cells) rule cannot give a value yet; the pixel rule can
    m = _manifest(run_cofactors(cfg, max_pixels=200_000).read_text())
    assert m["level"] == "pixels" and m["is_subtracted"] and all(x["suggested"] is None for x in m["markers"])
    m = _manifest(run_cofactors(cfg, max_pixels=200_000, subtracted="p1-pixels").read_text())
    assert {p["method"] for x in m["markers"] for p in x["per_sample"]} == {"p1-pixels"}
    # after segmentation, default rule: 1st percentile of the positive cell values, for every marker
    run_segmentation(cfg)
    m = _manifest(run_cofactors(cfg).read_text())
    assert m["level"] == "cells" and m["is_subtracted"]
    t = per_sample()
    assert set(t["method"]) == {"p1-cells"}
    cells = pd.concat([pd.read_parquet(cfg.sample_dir(s) / "cells.parquet") for s in ("Biopsy_A", "Biopsy_B")])
    for _, r in t.iterrows():
        v = cells.loc[cells["sample"] == r["sample"], f"{r['marker']}:cell:mean"].to_numpy()
        assert r["cofactor"] == sig2(np.percentile(v[v > 0], 1)), r
    # pixel rule after segmentation: reads the pixels; pixel values are larger than cell means
    run_cofactors(cfg, subtracted="p1-pixels")
    t2 = per_sample()
    assert set(t2["method"]) == {"p1-pixels"} and (t2["cofactor"].to_numpy() > t["cofactor"].to_numpy()).mean() > 0.5
