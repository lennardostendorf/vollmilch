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

from vollmilch.cofactors import run_cofactors, suggest  # noqa: E402
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


def test_split_background_subtracted_cells():
    """Background set to 0: zero cells, stray-signal cells (~10) and positive cells (~3000)."""
    r = np.random.default_rng(3)
    cells = np.r_[np.zeros(5000), r.lognormal(np.log(10), .8, 2500), r.lognormal(np.log(3000), .6, 2500)]
    s = suggest(cells, "cells")
    assert s["background_removed"] and s["method"] == "split", s
    assert 100 <= s["cofactor"] <= 400 and s["split_eta"] > 0.75, s  # between the humps (geometric middle ~170)
    # one hump only (marker on most cells): still a split, but flagged by its low separation
    s = suggest(np.r_[np.zeros(100), r.lognormal(np.log(3000), .6, 5000)], "cells")
    assert s["method"] == "split" and s["split_eta"] < 0.66, s
    # too few non-zero cells: no suggestion
    s = suggest(np.r_[np.zeros(1000), r.lognormal(3, 1, 10)], "cells")
    assert s["method"] == "none" and np.isnan(s["cofactor"])


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
