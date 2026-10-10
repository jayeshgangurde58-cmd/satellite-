"""Synthetic geometry tests; these validate grid logic, not satellite prediction accuracy."""
import json
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from quality_gate.aoi_coverage import aoi_mask_and_report


def _raster(path: Path):
    arr=np.ones((1,100,100),dtype=np.uint16)
    arr[:,0:10,:]=0
    with rasterio.open(path,"w",driver="GTiff",width=100,height=100,count=1,dtype="uint16",
                       crs="EPSG:4326",transform=from_origin(0,1,0.01,0.01),nodata=0) as dst:
        dst.write(arr)
    return path


def _geojson(path: Path, coords):
    path.write_text(json.dumps({"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":coords}}),encoding="utf-8")
    return path


def test_aoi_full_and_partial_coverage_and_valid_pixels(tmp_path):
    raster=_raster(tmp_path/"scene.tif")
    full=_geojson(tmp_path/"full.geojson",[[[0,1],[1,1],[1,0],[0,0],[0,1]]])
    mask, report=aoi_mask_and_report(full,raster)
    assert report["status"]=="PASS"
    assert mask.shape==(100,100) and mask.any()
    assert report["valid_fraction_within_rasterized_aoi"] < 1.0  # NoData top strip
    partial=_geojson(tmp_path/"partial.geojson",[[[0.8,1],[1.2,1],[1.2,0],[0.8,0],[0.8,1]]])
    _, rep=aoi_mask_and_report(partial,raster)
    assert rep["status"]=="WARNING"
    assert 0 < rep["aoi_footprint_coverage_fraction"] < 1


def test_aoi_outside_raster_rejected(tmp_path):
    raster=_raster(tmp_path/"scene.tif")
    aoi=_geojson(tmp_path/"outside.geojson",[[[5,5],[6,5],[6,4],[5,4],[5,5]]])
    mask, report=aoi_mask_and_report(aoi,raster)
    assert not mask.any()
    assert report["status"]=="REJECT"


def test_invalid_geojson_rejected(tmp_path):
    raster=_raster(tmp_path/"scene.tif")
    path=tmp_path/"bad.geojson"; path.write_text("not json",encoding="utf-8")
    with pytest.raises(ValueError):
        aoi_mask_and_report(path,raster)
