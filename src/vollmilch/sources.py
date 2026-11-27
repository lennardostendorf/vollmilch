"""Image sources. Every source exposes channel names, a pixel size and one or more
images that can be read one channel and one bounding box at a time.

    qptiff      Akoya/PerkinElmer .qptiff (marker names from the page descriptions)
    ome_tiff    one .ome.tif file, or a folder of them (one image each)
    png_folder  one PNG per channel (file name = marker name); one folder = one image,
                or a folder of such folders
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

log = logging.getLogger("vollmilch")


@dataclass
class ImageInfo:
    name: str
    height: int
    width: int
    path: str  # file to open in QuPath


def _dedupe(names):
    seen, out = {}, []
    for n in names:
        n = (n or "").strip() or "channel"
        if n in seen:
            seen[n] += 1
            n2 = f"{n}_{seen[n]}"
            log.warning("duplicate channel name %s renamed to %s", n, n2)
            out.append(n2)
        else:
            seen[n] = 1
            out.append(n)
    return out


class Source:
    channel_names: list[str]
    pixel_size_um: float
    images: list[ImageInfo]

    def channel_index(self, name: str) -> int:
        try:
            return self.channel_names.index(name)
        except ValueError:
            lower = [c.lower() for c in self.channel_names]
            if name.lower() in lower:
                return lower.index(name.lower())
            raise KeyError(f"channel '{name}' not found. Available: {', '.join(self.channel_names)}") from None

    def image(self, name: str) -> ImageInfo:
        for im in self.images:
            if im.name == name:
                return im
        raise KeyError(name)

    def read(self, image: str, channel: str | int, bbox=None) -> np.ndarray:
        """bbox = (y0, y1, x0, x1) in image pixels; returns a 2-D array."""
        raise NotImplementedError

    def close(self):
        pass


# ---------------------------------------------------------------------------- TIFF
class _TiffImage:
    """Level-0 array of one TIFF series, read lazily through zarr where possible."""

    def __init__(self, path: str):
        self.path = path
        self.tf = tifffile.TiffFile(path)
        self.series = self.tf.series[0]
        axes = self.series.axes.upper()
        shape = self.series.shape
        # keep C, Y, X; other axes must be singleton
        self.axes = axes
        for ax, n in zip(axes, shape):
            if ax not in "CYXS" and n != 1:
                raise ValueError(f"{path}: axis {ax} has size {n}; only 2-D multichannel images are supported")
        self.shape = dict(zip(axes, shape))
        self.height, self.width = self.shape["Y"], self.shape["X"]
        self.n_channels = self.shape.get("C", 1) if "C" in axes else self.shape.get("S", 1)
        self._z = None
        try:
            import zarr

            store = self.series.aszarr(level=0)
            z = zarr.open(store, mode="r")
            if hasattr(z, "shape"):
                self._z = z
            else:  # group
                self._z = z["0"]
        except Exception as e:  # pragma: no cover - depends on installed zarr
            log.info("zarr access unavailable (%s); reading whole pages", e)

    def read(self, c: int, bbox=None) -> np.ndarray:
        y0, y1, x0, x1 = bbox if bbox is not None else (0, self.height, 0, self.width)
        if self._z is not None:
            idx = []
            for ax in self.axes:
                if ax == "Y":
                    idx.append(slice(y0, y1))
                elif ax == "X":
                    idx.append(slice(x0, x1))
                elif ax in "CS" and ax == ("C" if "C" in self.axes else "S"):
                    idx.append(c)
                else:
                    idx.append(0)
            return np.asarray(self._z[tuple(idx)])
        # fallback: page-wise (one page per channel in planar files)
        if "C" in self.axes and len(self.series.pages) == self.n_channels:
            arr = self.series.pages[c].asarray()
            return arr[y0:y1, x0:x1]
        arr = self.series.asarray()
        sl = []
        for ax in self.axes:
            sl.append(slice(y0, y1) if ax == "Y" else slice(x0, x1) if ax == "X" else c if ax in "CS" else 0)
        return arr[tuple(sl)]

    def close(self):
        self.tf.close()


def _tiff_pixel_size(page) -> float | None:
    try:
        tags = page.tags
        xr = tags["XResolution"].value
        unit = tags["ResolutionUnit"].value
        ppu = xr[0] / xr[1] if isinstance(xr, tuple) else float(xr)
        if ppu <= 0:
            return None
        unit = int(unit)
        if unit == 3:  # cm
            return 1e4 / ppu
        if unit == 2:  # inch
            return 25400.0 / ppu
    except Exception:
        return None
    return None


class QptiffSource(Source):
    def __init__(self, path: str, pixel_size_um=None, channel_names=None):
        self.path = path
        self.t = _TiffImage(path)
        names = []
        for p in self.t.series.pages[: self.t.n_channels]:
            d = getattr(p, "description", "") or ""
            m = re.search(r"<Biomarker>(.*?)</Biomarker>", d, re.S)
            if not (m and m.group(1).strip()):
                m = re.search(r"<Name>(.*?)</Name>", d, re.S)
            names.append(m.group(1).strip() if m else f"ch{len(names)}")
        if channel_names:
            if len(channel_names) != self.t.n_channels:
                raise ValueError(f"channel_names has {len(channel_names)} entries, image has {self.t.n_channels} channels")
            names = list(channel_names)
        self.channel_names = _dedupe(names)
        px = pixel_size_um or _tiff_pixel_size(self.t.series.pages[0])
        if px is None:
            raise ValueError("pixel size not found in the qptiff; set source.pixel_size_um")
        self.pixel_size_um = float(px)
        self.images = [ImageInfo(Path(path).stem, self.t.height, self.t.width, path)]

    def read(self, image, channel, bbox=None):
        c = channel if isinstance(channel, int) else self.channel_index(channel)
        return self.t.read(c, bbox)

    def close(self):
        self.t.close()


class OmeTiffSource(Source):
    def __init__(self, path: str, pixel_size_um=None, channel_names=None):
        p = Path(path)
        files = sorted([f for f in p.iterdir() if f.name.lower().endswith((".ome.tif", ".ome.tiff", ".tif", ".tiff"))]) if p.is_dir() else [p]
        if not files:
            raise FileNotFoundError(f"no TIFF files in {path}")
        self.t = {}
        self.images = []
        names0, px0 = None, None
        for f in files:
            t = _TiffImage(str(f))
            name = re.sub(r"\.ome$", "", f.stem)
            self.t[name] = t
            self.images.append(ImageInfo(name, t.height, t.width, str(f)))
            names, px = None, None
            ome = t.tf.ome_metadata
            if ome:
                names = re.findall(r"<Channel\b[^>]*?\bName=\"([^\"]*)\"", ome)
                m = re.search(r"PhysicalSizeX=\"([0-9.eE+-]+)\"", ome)
                if m:
                    px = float(m.group(1))
                    mu = re.search(r"PhysicalSizeXUnit=\"([^\"]+)\"", ome)
                    if mu and mu.group(1) in ("nm",):
                        px /= 1000.0
                    elif mu and mu.group(1) in ("mm",):
                        px *= 1000.0
            if not names or len(names) != t.n_channels:
                names = [f"ch{i}" for i in range(t.n_channels)]
            px = px or _tiff_pixel_size(t.series.pages[0])
            if names0 is None:
                names0, px0 = names, px
            elif names != names0:
                raise ValueError(f"{f}: channel names differ from {files[0]}")
        if channel_names:
            if len(channel_names) != len(names0):
                raise ValueError("channel_names length does not match the image")
            names0 = list(channel_names)
        self.channel_names = _dedupe(names0)
        px = pixel_size_um or px0
        if px is None:
            raise ValueError("pixel size not found; set source.pixel_size_um")
        self.pixel_size_um = float(px)

    def read(self, image, channel, bbox=None):
        c = channel if isinstance(channel, int) else self.channel_index(channel)
        return self.t[image].read(c, bbox)

    def close(self):
        for t in self.t.values():
            t.close()


# ---------------------------------------------------------------------------- PNG folders
def _load_png(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        if im.mode in ("RGB", "RGBA", "P", "LA"):
            arr = np.asarray(im.convert("RGB"))
            if not (np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2])):
                log.warning("%s is a colour PNG; using the maximum over R, G, B", path.name)
            arr = arr.max(axis=2)
        else:
            arr = np.asarray(im)
    if arr.dtype == np.int32 or arr.dtype == np.int64:
        arr = arr.astype(np.uint16) if arr.min() >= 0 and arr.max() < 65536 else arr.astype(np.float32)
    elif arr.dtype == bool:
        arr = arr.astype(np.uint8)
    return arr


class PngFolderSource(Source):
    EXT = (".png", ".tif", ".tiff")

    def __init__(self, path: str, pixel_size_um=None, channel_name_regex=None, channel_names=None, image_dir=None):
        root = Path(path)
        if not root.is_dir():
            raise NotADirectoryError(path)

        def chan_files(d: Path):
            return sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in self.EXT and not f.name.startswith("."))

        direct = chan_files(root)
        folders = [root] if direct else sorted(d for d in root.iterdir() if d.is_dir() and chan_files(d))
        if not folders:
            raise FileNotFoundError(f"no channel images in {path} or its subfolders")
        rx = re.compile(channel_name_regex) if channel_name_regex else None

        def marker(f: Path):
            stem = f.stem
            if rx:
                m = rx.search(stem)
                if m:
                    return m.group(1) if m.groups() else m.group(0)
            return stem

        self._files: dict[str, dict[str, Path]] = {}
        sets = []
        for d in folders:
            fm = {}
            for f in chan_files(d):
                fm[marker(f)] = f
            self._files[d.name] = fm
            sets.append(list(fm))
        common = [c for c in sets[0] if all(c in s for s in sets)]
        for name, s in zip(self._files, sets):
            missing = set(sets[0]) ^ set(s)
            if missing:
                log.warning("image %s: channels %s not present in every image; ignored", name, sorted(missing))
        # nuclear stain first if present, rest alphabetical: stable, readable order
        common = sorted(common, key=lambda c: (c.upper() not in ("DAPI", "HOECHST", "DNA", "DNA1"), c.lower()))
        if channel_names:
            raise ValueError("channel_names cannot be used with png_folder; rename files or use channel_name_regex")
        self.channel_names = common
        if pixel_size_um is None:
            raise ValueError("source.pixel_size_um is required for png_folder")
        self.pixel_size_um = float(pixel_size_um)
        self._cache: dict[str, dict[str, np.ndarray]] = {}
        self.images = []
        self.image_dir = Path(image_dir) if image_dir else None
        for name, fm in self._files.items():
            a = _load_png(fm[common[0]])
            self.images.append(ImageInfo(name, a.shape[0], a.shape[1],
                                         str(self.image_dir / f"{name}.ome.tif") if self.image_dir else ""))

    def _get(self, image, ch):
        cache = self._cache.setdefault(image, {})
        if ch not in cache:
            if len(self._cache) > 2:  # keep memory bounded when iterating over many images
                for k in list(self._cache)[:-2]:
                    if k != image:
                        del self._cache[k]
            a = _load_png(self._files[image][ch])
            im = self.image(image)
            if a.shape != (im.height, im.width):
                raise ValueError(f"{self._files[image][ch]}: shape {a.shape} differs from {(im.height, im.width)}")
            cache[ch] = a
        return cache[ch]

    def read(self, image, channel, bbox=None):
        ch = self.channel_names[channel] if isinstance(channel, int) else self.channel_names[self.channel_index(channel)]
        a = self._get(image, ch)
        if bbox is None:
            return a
        y0, y1, x0, x1 = bbox
        return a[y0:y1, x0:x1]

    def write_ome_tiff(self, image: str, out: Path):
        """Stack the channel images into one OME-TIFF that QuPath opens with marker names."""
        out.parent.mkdir(parents=True, exist_ok=True)
        stack = np.stack([self.read(image, c) for c in self.channel_names])
        if stack.dtype not in (np.uint8, np.uint16, np.float32):
            stack = stack.astype(np.float32)
        tifffile.imwrite(
            out, stack, photometric="minisblack", compression="zlib", ome=True,
            metadata={"axes": "CYX", "Channel": {"Name": self.channel_names},
                      "PhysicalSizeX": self.pixel_size_um, "PhysicalSizeY": self.pixel_size_um,
                      "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm"},
        )


def open_source(cfg) -> Source:
    s = cfg["source"]
    t = s["type"]
    if t == "qptiff":
        return QptiffSource(s["path"], s.get("pixel_size_um"), s.get("channel_names"))
    if t == "ome_tiff":
        return OmeTiffSource(s["path"], s.get("pixel_size_um"), s.get("channel_names"))
    if t == "png_folder":
        return PngFolderSource(s["path"], s.get("pixel_size_um"), s.get("channel_name_regex"),
                               s.get("channel_names"), image_dir=Path(cfg["output"]) / "images")
    raise ValueError(f"unknown source.type {t!r} (qptiff, ome_tiff, png_folder)")
