"""Configuration: YAML file merged over the defaults in config_template.yaml."""
from __future__ import annotations

import copy
import logging
import os
from importlib import resources
from pathlib import Path

import yaml

log = logging.getLogger("vollmilch")

TEMPLATE = resources.files("vollmilch").joinpath("config_template.yaml")


def template_text() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


def _defaults() -> dict:
    d = yaml.safe_load(template_text())
    d["regions"] = {"type": "none", "path": None, "name_property": "name", "include": None}
    d["source"] = {"type": None, "path": None, "pixel_size_um": None,
                   "channel_name_regex": None, "channel_names": None}
    d["samples_table"] = None
    return d


def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in base:
            log.warning("unknown config key %s%s (ignored? check spelling)", path, k)
            out[k] = v
        elif isinstance(base[k], dict) and isinstance(v, dict) and k not in ("cofactors",):
            out[k] = _merge(base[k], v, f"{path}{k}.")
        else:
            out[k] = v
    return out


class Config(dict):
    """dict with attribute access for the top-level sections."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    @property
    def out(self) -> Path:
        return Path(self["output"])

    def sample_dir(self, sample: str) -> Path:
        return self.out / "samples" / sample


def read_cofactor_table(path) -> dict[str, float]:
    """marker -> cofactor from a CSV/TSV with columns marker, cofactor (others ignored, e.g. the
    file written by the cofactor viewer). Rows with an empty cofactor fall back to the default."""
    import pandas as pd

    t = pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)
    cols = {c.strip().lower(): c for c in t.columns}
    if "marker" not in cols or "cofactor" not in cols:
        raise ValueError(f"{path}: needs the columns marker and cofactor (found {', '.join(t.columns)})")
    out = {}
    for m, v in zip(t[cols["marker"]], t[cols["cofactor"]]):
        m, v = (m or "").strip(), (v or "").strip()
        if not m or not v:
            continue
        try:
            c = float(v)
        except ValueError:
            raise ValueError(f"{path}: cofactor of {m} is not a number: {v!r}") from None
        if not c > 0:
            raise ValueError(f"{path}: cofactor of {m} must be > 0 (is {v})")
        if m in out:
            raise ValueError(f"{path}: marker {m} listed twice")
        out[m] = c
    return out


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    user = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = Config(_merge(_defaults(), user))
    base = path.parent

    def res(p):
        if p is None:
            return None
        p = Path(str(p)).expanduser()
        return os.path.normpath(p if p.is_absolute() else base / p)

    if not cfg["source"].get("path") or not cfg["source"].get("type"):
        raise ValueError("config: source.type and source.path are required")
    cfg["source"]["path"] = res(cfg["source"]["path"])
    if cfg["regions"].get("path"):
        cfg["regions"]["path"] = res(cfg["regions"]["path"])
    if cfg.get("samples_table"):
        cfg["samples_table"] = res(cfg["samples_table"])
    cfg["output"] = res(cfg["output"])
    cfg["_config_path"] = str(path)
    tr = cfg["transform"]
    if isinstance(tr.get("cofactors"), str):
        tr["_cofactors_file"] = res(tr["cofactors"])
        tr["cofactors"] = read_cofactor_table(tr["_cofactors_file"])
    elif tr.get("cofactors") is not None and not isinstance(tr["cofactors"], dict):
        raise ValueError("transform.cofactors must be a mapping {marker: cofactor} or a CSV file")

    seg = cfg["segmentation"]
    if seg["mode"] not in ("nuclei", "expand", "cells"):
        raise ValueError("segmentation.mode must be nuclei, expand or cells")
    if seg["mode"] == "cells" and not seg["cyto_channels"]:
        raise ValueError("segmentation.mode=cells needs segmentation.cyto_channels")
    if seg["mode"] == "cells" and seg["backend"] != "cellpose":
        raise ValueError("segmentation.mode=cells needs backend=cellpose")
    for c in cfg["features"]["compartments"]:
        if c not in ("nucleus", "cell", "cytoplasm"):
            raise ValueError(f"unknown compartment {c}")
    for s in cfg["features"]["stats"]:
        if s not in ("median", "mean", "max", "p90", "sd"):
            raise ValueError(f"unknown stat {s}")
    if cfg["embedding"]["stat"] not in cfg["features"]["stats"]:
        raise ValueError(f"embedding.stat {cfg['embedding']['stat']} is not in features.stats {cfg['features']['stats']}")
    missing = [s for s in cfg["report"]["stats"] if s not in cfg["features"]["stats"]]
    if missing:
        log.warning("report.stats %s are not in features.stats %s and are left out", missing, cfg["features"]["stats"])
    return cfg
