"""vollmilch init: every --type and the MELC preset write a config that loads with the intended values."""
import pytest

from vollmilch.cli import main
from vollmilch.config import load_config


@pytest.mark.parametrize("args, expect", [
    ([], {"type": "qptiff", "px": None, "regions": "geojson", "comps": ["nucleus", "cell"]}),
    (["--type", "ome_tiff"], {"type": "ome_tiff", "px": None, "regions": "geojson", "comps": ["nucleus", "cell"]}),
    (["--type", "png_folder"], {"type": "png_folder", "px": 0.5, "regions": "none", "comps": ["nucleus", "cell"]}),
    (["--melc"], {"type": "png_folder", "px": 0.325, "regions": "none", "comps": ["cell"]}),
    (["--melc", "--type", "ome_tiff"], {"type": "ome_tiff", "px": 0.325, "regions": "geojson", "comps": ["cell"]}),
])
def test_init(tmp_path, args, expect):
    p = tmp_path / "config.yaml"
    main(["init", "-o", str(p), *args])
    cfg = load_config(p)
    assert cfg["source"]["type"] == expect["type"]
    assert cfg["source"]["pixel_size_um"] == expect["px"]
    assert cfg["regions"]["type"] == expect["regions"]
    assert cfg["features"]["compartments"] == expect["comps"]
    assert str(tmp_path / "data") in cfg["source"]["path"]
    if "--melc" in args:
        assert cfg["report"]["compartments"] == ["cell"] and "MELC preset" in p.read_text()
