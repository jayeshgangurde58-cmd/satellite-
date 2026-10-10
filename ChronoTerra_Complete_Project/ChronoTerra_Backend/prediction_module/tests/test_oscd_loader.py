from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image
from evaluation.oscd import load_oscd_pairs, read_oscd_pair, BANDS


def _write_tif(path: Path, array: np.ndarray):
    profile = {"driver": "GTiff", "height": array.shape[0], "width": array.shape[1],
               "count": 1, "dtype": "uint16", "transform": from_origin(0, 10, 1, 1), "crs": "EPSG:32643"}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.uint16), 1)


def test_reads_official_style_oscd_layout(tmp_path):
    images = tmp_path / "Onera Satellite Change Detection dataset - Images" / "TESTCITY"
    labels = tmp_path / "Onera Satellite Change Detection dataset - Test Labels" / "TESTCITY" / "cm"
    (images / "imgs_1_rect").mkdir(parents=True)
    (images / "imgs_2_rect").mkdir(parents=True)
    labels.mkdir(parents=True)
    for i, band in enumerate(BANDS):
        arr = np.full((8, 9), i + 10, dtype=np.uint16)
        _write_tif(images / "imgs_1_rect" / f"TESTCITY_{band}.tif", arr)
        _write_tif(images / "imgs_2_rect" / f"TESTCITY_{band}.tif", arr + 1)
    mask = np.zeros((8, 9), dtype=np.uint8)
    mask[2:4, 3:6] = 255
    Image.fromarray(mask).save(labels / "cm.png")

    pairs = load_oscd_pairs(tmp_path, "test")
    assert len(pairs) == 1 and pairs[0].name == "TESTCITY"
    before, after, truth, valid = read_oscd_pair(pairs[0])
    assert before.shape == after.shape == (13, 8, 9)
    assert truth.sum() == 6
    assert valid.all()


def test_missing_oscd_root_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError, match="Could not find directory"):
        load_oscd_pairs(tmp_path, "test")


def test_oscd_scale_audit_uses_all_development_scenes_and_rejects_mixed_ranges(tmp_path):
    import pytest
    from models.oscd_data import OSCDPairFiles, audit_scale_factor_across_scenes

    def make_scene(name, scale):
        before=[]; after=[]
        for role, target in [('b', before), ('a', after)]:
            folder=tmp_path/name/role; folder.mkdir(parents=True, exist_ok=True)
            for band in BANDS:
                path=folder/f'{name}_{band}.tif'
                val=1000 if scale==10000 else 0.1
                _write_tif(path, np.full((16, 16), val, dtype=np.uint16 if scale==10000 else np.float32))
                target.append(path)
        label=tmp_path/name/'cm.png'; Image.fromarray(np.zeros((16,16),dtype=np.uint8)).save(label)
        return OSCDPairFiles(name, tuple(before), tuple(after), label)

    s1=make_scene('dn_scene',10000); s2=make_scene('dn_scene2',10000)
    factor,audit=audit_scale_factor_across_scenes([s1,s2])
    assert factor==10000.0 and set(audit)=={'dn_scene','dn_scene2'}
    s3=make_scene('unit_scene',1)
    with pytest.raises(ValueError, match='mix raw-DN and unit-reflectance'):
        audit_scale_factor_across_scenes([s1,s3])
