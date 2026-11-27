"""Regions (samples): QuPath GeoJSON annotations, a label/binary mask image, or whole images."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("vollmilch")


def safe_name(s: str) -> str:
    s = re.sub(r"[^\w.\-+]+", "_", str(s).strip())
    return s.strip("_") or "sample"


@dataclass
class Region:
    sample: str
    image: str
    name: str
    bbox: tuple  # (y0, y1, x0, x1) in image pixels
    geometry: object = None  # shapely geometry in image pixels (GeoJSON regions)
    label_path: str | None = None
    label_value: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        y0, y1, x0, x1 = self.bbox
        return (y1 - y0, x1 - x0)

    def mask(self) -> np.ndarray:
        """Boolean mask of the region inside its bounding box."""
        y0, y1, x0, x1 = self.bbox
        h, w = y1 - y0, x1 - x0
        if self.geometry is not None:
            m = np.zeros((h, w), np.uint8)
            polys = list(self.geometry.geoms) if self.geometry.geom_type == "MultiPolygon" else [self.geometry]
            for p in polys:
                rings = [p.exterior, *p.interiors]
                pts = [np.round((np.asarray(r.coords)[:, :2] - 0.5 - (x0, y0)) * 16).astype(np.int32) for r in rings]
                cv2.fillPoly(m, pts, 1, lineType=cv2.LINE_8, shift=4)
            return m.astype(bool)
        if self.label_path is not None:
            lab = _read_label(self.label_path)
            return lab[y0:y1, x0:x1] == self.label_value
        return np.ones((h, w), bool)

    def to_json(self):
        d = {"sample": self.sample, "image": self.image, "name": self.name, "bbox": list(map(int, self.bbox))}
        if self.geometry is not None:
            from shapely.geometry import mapping

            d["geometry"] = mapping(self.geometry)
        return d


_label_cache: dict = {}


def _read_label(path: str) -> np.ndarray:
    if path not in _label_cache:
        _label_cache.clear()
        if path.lower().endswith((".tif", ".tiff")):
            import tifffile

            a = tifffile.imread(path)
        else:
            from PIL import Image

            with Image.open(path) as im:
                a = np.asarray(im)
        if a.ndim == 3:
            a = a.max(axis=2)
        _label_cache[path] = a
    return _label_cache[path]


def _feature_name(props: dict, key: str):
    props = props or {}
    v = props.get(key)
    if v:
        return str(v)
    if props.get("name"):
        return str(props["name"])
    cls = props.get("classification")
    if isinstance(cls, dict) and cls.get("name"):
        return str(cls["name"])
    if isinstance(cls, str) and cls:
        return cls
    return None


def read_geojson_regions(path: str, name_property="name"):
    from shapely.geometry import shape
    from shapely.validation import make_valid

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    feats = data["features"] if isinstance(data, dict) and "features" in data else data if isinstance(data, list) else [data]
    out = []
    for i, f in enumerate(feats):
        g = f.get("geometry")
        if not g or g.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        geom = shape(g)
        if not geom.is_valid:
            geom = make_valid(geom)
            if geom.geom_type == "GeometryCollection":
                from shapely.geometry import MultiPolygon

                geom = MultiPolygon([p for p in geom.geoms if p.geom_type == "Polygon"])
        name = _feature_name(f.get("properties"), name_property) or f"region_{i + 1}"
        out.append((name, geom))
    if not out:
        raise ValueError(f"{path}: no polygon annotations found")
    return out


def get_regions(cfg, source) -> list[Region]:
    rc = cfg["regions"]
    rtype = rc.get("type") or "none"
    multi = len(source.images) > 1
    regions: list[Region] = []
    for im in source.images:
        def full():
            return Region(safe_name(im.name), im.name, im.name, (0, im.height, 0, im.width))

        if rtype == "none":
            regions.append(full())
            continue
        tmpl = rc["path"]
        if "{image}" in tmpl:
            p = tmpl.replace("{image}", im.name)
        else:
            if multi:
                raise ValueError("regions.path must contain {image} when the source has several images")
            p = tmpl
        if not Path(p).exists():
            if multi:
                log.warning("no region file %s; analysing image %s as a whole", p, im.name)
                regions.append(full())
                continue
            raise FileNotFoundError(p)
        if rtype == "geojson":
            items = read_geojson_regions(p, rc.get("name_property", "name"))
            names_seen = {}
            for name, geom in items:
                k = names_seen.get(name, 0)
                names_seen[name] = k + 1
                uname = name if k == 0 else f"{name}_{k + 1}"
                if k:
                    log.warning("duplicate region name %s renamed %s", name, uname)
                minx, miny, maxx, maxy = geom.bounds
                bbox = (max(0, int(np.floor(miny)) - 1), min(im.height, int(np.ceil(maxy)) + 1),
                        max(0, int(np.floor(minx)) - 1), min(im.width, int(np.ceil(maxx)) + 1))
                if bbox[1] <= bbox[0] or bbox[3] <= bbox[2]:
                    log.warning("region %s lies outside image %s; skipped", uname, im.name)
                    continue
                sample = safe_name(f"{im.name}_{uname}" if multi else uname)
                regions.append(Region(sample, im.name, uname, bbox, geometry=geom))
        elif rtype == "label_image":
            lab = _read_label(p)
            if lab.shape[:2] != (im.height, im.width):
                raise ValueError(f"{p}: mask shape {lab.shape[:2]} differs from image {(im.height, im.width)}")
            vals = [int(v) for v in np.unique(lab) if v != 0]
            for v in vals:
                ys, xs = np.nonzero(lab == v)
                bbox = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
                name = im.name if len(vals) == 1 else f"r{v}"
                sample = safe_name(name if len(vals) == 1 or not multi else f"{im.name}_{name}")
                if len(vals) > 1 and not multi:
                    sample = safe_name(name)
                regions.append(Region(sample, im.name, name, bbox, label_path=p, label_value=v))
        else:
            raise ValueError(f"unknown regions.type {rtype!r} (geojson, label_image, none)")
    inc = rc.get("include")
    if inc:
        inc = set(map(str, inc))
        regions = [r for r in regions if r.name in inc or r.sample in inc]
    dup = {r.sample for r in regions if [x.sample for x in regions].count(r.sample) > 1}
    if dup:
        raise ValueError(f"duplicate sample names: {sorted(dup)}")
    return regions
