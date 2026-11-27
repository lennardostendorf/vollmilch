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
        if s not in ("mean", "max", "p90", "sd"):
            raise ValueError(f"unknown stat {s}")
    return cfg
