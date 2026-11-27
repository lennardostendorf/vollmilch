"""Command line interface:  vollmilch <command> config.yaml ..."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__

log = logging.getLogger("vollmilch")


def _cfg(path):
    from .config import load_config

    return load_config(path)


PX_LINE = "  # pixel_size_um: 0.5          # required for png_folder; overrides file metadata otherwise\n"
MELC_PIXEL_SIZE_UM = 0.325


def _sub(txt: str, old: str, new: str) -> str:
    """Replace one template line; fail loudly if the template changed, rather than write a
    half-edited config."""
    if old not in txt:
        raise RuntimeError(f"config template has no {old.strip()!r}; update cmd_init")
    return txt.replace(old, new, 1)


def cmd_init(a):
    from .config import template_text

    out = Path(a.output)
    if out.exists() and not a.force:
        sys.exit(f"{out} exists (use --force to overwrite)")
    kind = a.type or ("png_folder" if a.melc else "qptiff")
    txt = template_text()
    px_line = PX_LINE
    if kind == "png_folder":
        px_line = "  pixel_size_um: 0.5            # required for png_folder; overrides file metadata otherwise\n"
        txt = _sub(txt, "type: qptiff ", "type: png_folder")
        txt = _sub(txt, "path: data/slide.qptiff       # <- your image",
                   "path: data                    # folder of <marker>.png, or a folder of such folders")
        txt = _sub(txt, PX_LINE, px_line)
        txt = _sub(txt, "type: geojson                 # geojson", "type: none                    # geojson")
    elif kind == "ome_tiff":
        txt = _sub(txt, "type: qptiff ", "type: ome_tiff")
        txt = _sub(txt, "path: data/slide.qptiff       # <- your image",
                   "path: data/image.ome.tif      # <- your image, or a folder of .ome.tif files")
    if a.melc:  # MELC preset: known pixel size, markers measured over the whole cell
        txt = _sub(txt, "# vollmilch configuration\n", "# vollmilch configuration (MELC preset: vollmilch init --melc)\n")
        txt = _sub(txt, px_line, f"  pixel_size_um: {MELC_PIXEL_SIZE_UM}          # MELC; overrides file metadata\n")
        txt = _sub(txt, "  compartments: [nucleus, cell] # any of",
                   "  compartments: [cell]          # MELC preset; any of")
        txt = _sub(txt, "  compartments: [cell, nucleus] # marker compartments offered in the viewer",
                   "  compartments: [cell]          # marker compartments offered in the viewer")
    out.write_text(txt, encoding="utf-8")
    preset = f" (MELC preset: {kind}, {MELC_PIXEL_SIZE_UM} µm/px, cell compartment)" if a.melc else ""
    print(f"wrote {out}{preset}: set the image (source.path), the nuclear channel and the regions in it, then run: "
          f"vollmilch inspect {out}")


def cmd_inspect(a):
    from .regions import get_regions
    from .sources import open_source

    cfg = _cfg(a.config)
    src = open_source(cfg)
    print(f"source      {cfg['source']['type']}: {cfg['source']['path']}")
    print(f"pixel size  {src.pixel_size_um:.4f} µm/px")
    print(f"channels    ({len(src.channel_names)}) {', '.join(src.channel_names)}")
    print(f"images      {len(src.images)}")
    for im in src.images[:20]:
        print(f"  {im.name}: {im.width} x {im.height} px")
    regs = get_regions(cfg, src)
    print(f"samples     {len(regs)}")
    for r in regs:
        h, w = r.shape
        area = r.mask().sum() * src.pixel_size_um**2 / 1e6 if a.area else None
        print(f"  {r.sample:30s} image {r.image}  bbox {w} x {h} px" + (f"  {area:.2f} mm²" if area is not None else ""))
    seg = cfg["segmentation"]
    need = [seg["nuclear_channel"], *seg.get("cyto_channels", []), *(cfg["thumbnails"].get("channels") or [])]
    if cfg["features"]["channels"] not in ("all", None):
        need += list(cfg["features"]["channels"])
    bad = []
    for c in need:
        try:
            src.channel_index(c)
        except KeyError:
            bad.append(c)
    if bad:
        print(f"\nWARNING: channels not found in the source: {', '.join(sorted(set(bad)))}")
    else:
        print("\nall configured channels found")
    cof = cfg["transform"].get("cofactors") or {}
    if cof:
        unknown = [m for m in cof if m not in src.channel_names]
        f = cfg["transform"].get("_cofactors_file")
        print(f"cofactors   {len(cof)} markers" + (f" from {f}" if f else "") + f"; others use {cfg['transform'].get('cofactor', 5)}")
        if unknown:
            print(f"WARNING: cofactors for markers that are not channels (check spelling and case): {', '.join(unknown)}")
    src.close()


def cmd_segment(a):
    from .pipeline import run_segmentation, setup_logging

    cfg = _cfg(a.config)
    setup_logging(cfg.out, a.verbose)
    run_segmentation(cfg, a.samples.split(",") if a.samples else None, force=a.force, reuse_segmentation=a.reuse_segmentation)


def cmd_cofactors(a):
    from .cofactors import run_cofactors
    from .pipeline import setup_logging

    cfg = _cfg(a.config)
    setup_logging(cfg.out, a.verbose)
    print(run_cofactors(cfg, level=a.level, samples=a.samples.split(",") if a.samples else None,
                        max_pixels=a.max_pixels, out_path=a.output))


def cmd_embed(a):
    from .embed import run_embedding
    from .gating import evaluate_gates, load_embeddings, load_gates, viewer_features
    from .pipeline import load_cells, setup_logging
    from .regions import safe_name

    cfg = _cfg(a.config)
    setup_logging(cfg.out, a.verbose)
    df = load_cells(cfg)
    name = a.name or "all"
    if a.population:
        if not a.gates:
            sys.exit("--population needs --gates")
        vals, _, _ = viewer_features(df, cfg, load_embeddings(cfg))
        member, pops, _ = evaluate_gates(vals, load_gates(a.gates))
        sel = [p for p in pops if p["name"] == a.population]
        if not sel:
            sys.exit(f"population {a.population!r} not found; available: {[p['name'] for p in pops]}")
        df = df[member[sel[0]["id"]]].reset_index(drop=True)
        name = a.name or safe_name(a.population)
        log.info("population %s: %d cells", a.population, len(df))
    markers = a.markers.split(",") if a.markers else None
    run_embedding(cfg, df, name=name, markers=markers, batch_key=None if a.no_harmony else "config")
    if a.report:
        from .report import build_report

        build_report(cfg)


def cmd_report(a):
    from .pipeline import setup_logging
    from .report import build_report

    cfg = _cfg(a.config)
    setup_logging(None, a.verbose)
    p = build_report(cfg, a.output, gates_path=a.gates, title=a.title)
    print(p)


def cmd_export(a):
    from .export import export_gates
    from .pipeline import setup_logging

    cfg = _cfg(a.config)
    setup_logging(cfg.out, a.verbose)
    d = export_gates(cfg, a.gates, samples=a.samples.split(",") if a.samples else None, only=a.only,
                     measurements=a.measurements, full_path=a.full_path, out_dir=a.output, geometry=not a.no_geometry)
    print(d)


def cmd_to_seurat(a):
    from .pipeline import setup_logging
    from .seurat import to_seurat

    cfg = _cfg(a.config)
    setup_logging(cfg.out, a.verbose)
    print(to_seurat(cfg, out_dir=a.output, gates_path=a.gates, compartment=a.compartment, stat=a.stat,
                    assay=a.assay, samples=a.samples.split(",") if a.samples else None, rds=not a.no_rds))


def cmd_run(a):
    from .embed import run_embedding
    from .pipeline import load_cells, run_segmentation, setup_logging
    from .report import build_report

    cfg = _cfg(a.config)
    setup_logging(cfg.out, a.verbose)
    run_segmentation(cfg, force=a.force)
    if not a.no_embed:
        run_embedding(cfg, load_cells(cfg))
    print(build_report(cfg))


def main(argv=None):
    p = argparse.ArgumentParser("vollmilch", description="Multiplex histology: segmentation, features, HTML gating, QuPath export")
    p.add_argument("--version", action="version", version=f"vollmilch {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help, config=True):
        s = sub.add_parser(name, help=help, description=help)
        if config:
            s.add_argument("config", help="config YAML")
        s.add_argument("-v", "--verbose", action="store_true")
        s.set_defaults(fn=fn)
        return s

    s = add("init", cmd_init, "write a commented config template", config=False)
    s.add_argument("-o", "--output", default="config.yaml")
    s.add_argument("--type", choices=["qptiff", "png_folder", "ome_tiff"],
                   help="image source (default: qptiff, or png_folder with --melc)")
    s.add_argument("--melc", action="store_true",
                   help=f"MELC preset: png_folder (unless --type), {MELC_PIXEL_SIZE_UM} µm/px, markers measured in the cell compartment")
    s.add_argument("--force", action="store_true")

    s = add("inspect", cmd_inspect, "show channels, pixel size and samples; check the config")
    s.add_argument("--area", action="store_true", help="also compute region areas")

    s = add("segment", cmd_segment, "segment every sample and extract features + thumbnails (resumable)")
    s.add_argument("--samples", help="comma-separated sample names")
    s.add_argument("--force", action="store_true", help="redo finished samples")
    s.add_argument("--reuse-segmentation", action="store_true", help="keep label images; redo features/thumbnails")

    s = add("cofactors", cmd_cofactors, "suggest asinh cofactors per marker; HTML viewer with ridge plots to tune them")
    s.add_argument("--from", dest="level", choices=["auto", "pixels", "cells"], default="auto",
                   help="cells: the segmented cells' intensities (the values that are transformed); pixels: image "
                        "pixels inside the regions, before segmentation; auto (default): cells once every sample is segmented")
    s.add_argument("--samples", help="comma-separated sample names")
    s.add_argument("--max-pixels", type=int, default=2_000_000, help="pixels sampled per sample (random windows)")
    s.add_argument("-o", "--output", help="HTML path (default: <output>/cofactors/cofactors.html)")

    s = add("embed", cmd_embed, "UMAP + Leiden (uncorrected and Harmony) on all cells or a gated population")
    s.add_argument("--name", help="embedding name (default: all, or the population name)")
    s.add_argument("--gates", help="gates JSON saved from the viewer")
    s.add_argument("--population", help="embed only this population (needs --gates)")
    s.add_argument("--markers", help="comma-separated markers (default: embedding.markers)")
    s.add_argument("--no-harmony", action="store_true")
    s.add_argument("--report", action="store_true", help="rebuild the report afterwards")

    s = add("report", cmd_report, "build the HTML gating viewer")
    s.add_argument("-o", "--output", help="HTML path (default: <output>/report.html)")
    s.add_argument("--gates", help="embed these gates as the report's starting gates")
    s.add_argument("--title")

    s = add("export", cmd_export, "apply gates to all cells; write QuPath GeoJSON and class tables")
    s.add_argument("gates", help="gates JSON saved from the viewer")
    s.add_argument("--samples")
    s.add_argument("--only", help="write only cells of this population to the GeoJSON")
    s.add_argument("--measurements", choices=["none", "markers", "all"], default="markers")
    s.add_argument("--full-path", action="store_true", help="QuPath class = 'Parent: Child' (derived classes)")
    s.add_argument("--no-geometry", action="store_true", help="only the CSV tables")
    s.add_argument("-o", "--output", help="output folder (default: <output>/export/<gates name>)")

    s = add("to_seurat", cmd_to_seurat, "Seurat object (seurat.rds, if R with Seurat is installed) or the matrix files to build one")
    s.add_argument("--gates", help="gates JSON: adds population, population_path and gate_<name> to the metadata")
    s.add_argument("--compartment", help="default: embedding.compartment")
    s.add_argument("--stat", help="default: embedding.stat (must be in features.stats)")
    s.add_argument("--assay", default="Protein", help="Seurat assay name")
    s.add_argument("--samples")
    s.add_argument("--no-rds", action="store_true", help="only write the files, do not run R")
    s.add_argument("-o", "--output", help="output folder (default: <output>/seurat)")

    s = add("run", cmd_run, "segment + embed + report")
    s.add_argument("--force", action="store_true")
    s.add_argument("--no-embed", action="store_true")

    a = p.parse_args(argv)
    lg = logging.getLogger("vollmilch")
    lg.propagate = False
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        lg.addHandler(h)
    lg.setLevel(logging.DEBUG if getattr(a, "verbose", False) else logging.INFO)
    a.fn(a)


if __name__ == "__main__":
    main()
