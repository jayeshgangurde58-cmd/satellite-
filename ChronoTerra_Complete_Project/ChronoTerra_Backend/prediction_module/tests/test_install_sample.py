"""Tests for the Sentinel-2 L1C install script using synthetic local fixtures."""
import json
import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from tools import install_sample_l1c_pair

BANDS = install_sample_l1c_pair.BANDS

@pytest.fixture
def mock_l1c_server(tmp_path, monkeypatch):
    # Monkeypatch to avoid downloading real data and to allow small test windows
    monkeypatch.setattr(install_sample_l1c_pair, "TARGET_X_OFF", 0)
    monkeypatch.setattr(install_sample_l1c_pair, "TARGET_Y_OFF", 0)
    monkeypatch.setattr(install_sample_l1c_pair, "TARGET_SIZE", 32)
    
    server_dir = tmp_path / "server"
    before_dir = server_dir / "before"
    after_dir = server_dir / "after"
    before_dir.mkdir(parents=True)
    after_dir.mkdir(parents=True)
    
    # 10m transform
    transform_10m = from_origin(500000, 4600000, 10, 10)
    # 20m transform
    transform_20m = from_origin(500000, 4600000, 20, 20)
    
    for d, date_str in [(before_dir, "2023-05-22T00:00:00.000Z"), (after_dir, "2023-08-20T00:00:00.000Z")]:
        (d / "tileInfo.json").write_text(json.dumps({"timestamp": date_str}))
        for b in BANDS:
            # mock B05, B06, B07, B8A, B11, B12 as 20m
            if b in ("B05", "B06", "B07", "B8A", "B11", "B12"):
                w, h = 16, 16
                tf = transform_20m
            else:
                w, h = 32, 32
                tf = transform_10m
            
            with rasterio.open(d / f"{b}.jp2", "w", driver="GTiff", width=w, height=h,
                               count=1, dtype="uint16", crs="EPSG:32631", transform=tf) as dst:
                dst.write(np.full((1, h, w), int(b.replace("B", "").replace("A", "")), dtype="uint16"))

    # Patch urlopen to read local files if a local path is provided
    original_urlopen = urllib.request.urlopen
    def mock_urlopen(req, *args, **kwargs):
        url = req if isinstance(req, str) else req.full_url
        if os.path.exists(url) or (':' in url and not url.startswith('http')):
            path = Path(url)
            if not path.exists():
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
            return original_urlopen(f"file:///{path.resolve().as_posix()}", *args, **kwargs)
        return original_urlopen(req, *args, **kwargs)
        
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
    
    # Patch urlopen to handle raw windows paths and file://
    original_urlopen = urllib.request.urlopen
    def mock_urlopen(req, *args, **kwargs):
        url = req if isinstance(req, str) else req.full_url
        if url.startswith("file:///"):
            path = Path(url[8:])
        elif os.path.exists(url) or (':' in url and not url.startswith('http')):
            path = Path(url)
        else:
            return original_urlopen(req, *args, **kwargs)
            
        if not path.exists():
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return open(path, "rb")
        
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    return server_dir

def test_install_sample_success(tmp_path, mock_l1c_server, monkeypatch):
    out_dir = tmp_path / "out"
    before_url = str(mock_l1c_server / "before")
    after_url = str(mock_l1c_server / "after")
    args = ["--out-dir", str(out_dir), "--before-base", before_url, "--after-base", after_url]
    monkeypatch.setattr("sys.argv", ["install_sample_l1c_pair.py"] + args)
    
    install_sample_l1c_pair.main()
    
    assert (out_dir / "before.tif").is_file()
    assert (out_dir / "after.tif").is_file()
    assert (out_dir / "pair_metadata.json").is_file()
    
    # Check 13 bands and 10m target grid
    with rasterio.open(out_dir / "before.tif") as ds:
        assert ds.count == 13
        assert ds.width == 32 and ds.height == 32
        assert ds.crs.to_epsg() == 32631
        assert ds.transform.a == 10 and ds.transform.e == -10
        assert ds.descriptions == BANDS
        assert ds.tags()["ACQUISITION_DATE"] == "2023-05-22T00:00:00.000Z"
        
        # Verify B05 (20m resampled to 10m)
        b05 = ds.read(5)
        assert np.all(b05 == 5)
        
    manifest = json.loads((out_dir / "pair_metadata.json").read_text())
    assert manifest["before"]["timestamp"] == "2023-05-22T00:00:00.000Z"
    assert manifest["target_resolution_m"] == 10
    assert manifest["band_order"] == list(BANDS)

def test_install_sample_missing_band_fails_and_cleans_up(tmp_path, mock_l1c_server, monkeypatch):
    out_dir = tmp_path / "out"
    before_url = str(mock_l1c_server / "before")
    after_url = str(mock_l1c_server / "after")
    args = ["--out-dir", str(out_dir), "--before-base", before_url, "--after-base", after_url]
    monkeypatch.setattr("sys.argv", ["install_sample_l1c_pair.py"] + args)
    
    # delete a band to cause failure
    (mock_l1c_server / "before" / "B03.jp2").unlink()
    
    with pytest.raises(Exception):
        install_sample_l1c_pair.main()
        
    # tmp dir should be cleaned up
    assert not (out_dir / ".tmp_download").exists()
    assert not (out_dir / "before.tif").exists()

def test_install_sample_repeated_installation_overwrites(tmp_path, mock_l1c_server, monkeypatch):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "before.tif").write_bytes(b"old")
    (out_dir / "pair_metadata.json").write_text("old")
    
    before_url = str(mock_l1c_server / "before")
    after_url = str(mock_l1c_server / "after")
    args = ["--out-dir", str(out_dir), "--before-base", before_url, "--after-base", after_url]
    monkeypatch.setattr("sys.argv", ["install_sample_l1c_pair.py"] + args)
    install_sample_l1c_pair.main()
    
    assert (out_dir / "before.tif").read_bytes() != b"old"
    assert "old" not in (out_dir / "pair_metadata.json").read_text()

def test_install_sample_source_access_failure(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    args = ["--out-dir", str(out_dir), "--before-base", "https://invalid.domain.test/xyz", "--after-base", "https://invalid.domain.test/abc"]
    monkeypatch.setattr("sys.argv", ["install_sample_l1c_pair.py"] + args)
    
    with pytest.raises(Exception):
        install_sample_l1c_pair.main()
