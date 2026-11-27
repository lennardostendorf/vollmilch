"""Unit tests that need no Cellpose weights:  pytest tests/"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import make_synthetic as ms  # noqa: E402

from vollmilch.gating import evaluate_gates, in_polygon, round_sig  # noqa: E402
from vollmilch.segmentation import (WatershedBackend, finalize, match_nuclei_cells,  # noqa: E402
                                   segment_tiled, window_cores, window_starts)


@pytest.fixture(scope="module")
def tissue():
    rng = np.random.default_rng(3)
    m = np.ones((420, 420), bool)
    cs = ms.centers_in(m, 500, 15, rng)
    imgs, nuc, cell, truth = ms.paint(m.shape, cs, rng)
    return imgs, nuc, cell


def test_windows_partition():
    for n in (100, 2048, 5000, 7777):
        st = window_starts(n, 2048, 160)
        cores = window_cores(st, 2048, n)
        assert cores[0][0] == 0 and cores[-1][1] == n
        assert all(a[1] == b[0] for a, b in zip(cores, cores[1:]))
        for s, (c0, c1) in zip(st, cores):
            assert s <= c0 and c1 <= min(s + 2048, n)


def test_tiling_matches_whole_image(tissue):
    imgs, nuc, _ = tissue
    d = imgs["DAPI"].astype(np.float32)
    lo, hi = np.percentile(d, [1, 99.8])
    x = np.zeros(d.shape + (3,), np.float32)
    x[..., 0] = (d - lo) / (hi - lo)
    be = WatershedBackend({"diameter_um": 6}, 1.0)
    be.set_threshold(x[..., 0].ravel())
    whole = segment_tiled(lambda a, b, c, e: x[a:b, c:e], d.shape, None, be, tile=1000, overlap=60)
    tiled = segment_tiled(lambda a, b, c, e: x[a:b, c:e], d.shape, None, be, tile=150, overlap=60)
    n_w, n_t = whole.max(), tiled.max()
    assert abs(int(n_w) - int(n_t)) <= 0.03 * n_w
    # no object may be split at a seam: every tiled object overlaps one whole-image object mostly
    both = (whole > 0) & (tiled > 0)
    pairs = pd.DataFrame({"t": tiled[both], "w": whole[both]})
    frac = pairs.groupby("t")["w"].agg(lambda s: s.value_counts().iloc[0] / len(s))
    assert (frac > 0.8).mean() > 0.97


def test_match_nuclei_cells(tissue):
    _, nuc, cell = tissue
    rng = np.random.default_rng(0)
    # scramble the cell ids, drop 5 % of the cells so their nuclei become unmatched
    n = cell.max()
    perm = np.r_[0, rng.permutation(n) + 1]
    cl = perm[cell].astype(np.uint32)
    drop = rng.choice(np.arange(1, n + 1), n // 20, replace=False)
    cl[np.isin(cl, drop)] = 0
    nl, cl2 = match_nuclei_cells(nuc.astype(np.uint32), cl, None, 3, keep_unmatched=True)
    nl, cl2 = finalize(nl, cl2, None, 5, 10)
    # every nucleus pixel belongs to the cell with the same id
    m = nl > 0
    assert np.all(cl2[m] == nl[m])
    # nearly all original nuclei are kept (dropped cells are rebuilt by expansion)
    assert nl.max() >= 0.97 * n


def test_polygon_and_rounding():
    v = [[0, 0], [2, 0], [2, 2], [0, 2]]
    px = np.array([1, 3, np.nan, 0.5])
    py = np.array([1, 1, 1, 1.5])
    assert in_polygon(px, py, v).tolist() == [True, False, False, True]
    assert round_sig(np.array([123456.0, 0.00123456, -2.34567]))[0] == np.float32(123500)


def test_gate_hierarchy():
    vals = pd.DataFrame({"a": np.array([1, 5, 5, 9], np.float32), "b": np.array([1, 1, 9, 9], np.float32)})
    g = {"format": "vollmilch-gates", "populations": [
        {"id": "p", "name": "P", "parent": None, "x": "a", "y": "b", "type": "rect", "x0": 4, "x1": None, "y0": None, "y1": None},
        {"id": "c", "name": "C", "parent": "p", "x": "a", "y": "b", "type": "polygon", "vertices": [[0, 5], [10, 5], [10, 10], [0, 10]]},
    ]}
    member, pops, best = evaluate_gates(vals, g)
    assert member["p"].tolist() == [False, True, True, True]
    assert member["c"].tolist() == [False, False, True, True]
    assert best.tolist() == [-1, 0, 1, 1]
