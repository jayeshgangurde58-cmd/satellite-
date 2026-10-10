"""Generate an obviously synthetic GeoTIFF pair to test the app/UI. Not satellite data."""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "synthetic_demo"
OUT.mkdir(parents=True, exist_ok=True)
HEIGHT = WIDTH = 256
rng = np.random.default_rng(20261009)
y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
# Stylized urban background: vegetation, roads, and existing built blocks.
base = np.zeros((3, HEIGHT, WIDTH), dtype=np.float32)
base[0] = 70 + 12 * np.sin(x / 20) + rng.normal(0, 2, (HEIGHT, WIDTH))
base[1] = 100 + 10 * np.cos(y / 16) + rng.normal(0, 2, (HEIGHT, WIDTH))
base[2] = 65 + rng.normal(0, 2, (HEIGHT, WIDTH))
# Existing roofs and roads (identical in both dates).
for y0, x0, y1, x1 in [(25, 30, 55, 70), (80, 180, 115, 218), (165, 40, 205, 88)]:
    base[:, y0:y1, x0:x1] = np.array([155, 150, 140], dtype=np.float32)[:, None, None]
base[:, 125:140, :] = np.array([118, 118, 118], dtype=np.float32)[:, None, None]
base[:, :, 115:128] = np.array([115, 115, 115], dtype=np.float32)[:, None, None]
before = base.copy()
after = base.copy()
# A new built-up block: synthetic illustration only.
after[:, 48:88, 100:150] = np.array([190, 175, 160], dtype=np.float32)[:, None, None]
# Mild sensor-like noise on both dates.
before += rng.normal(0, 0.4, before.shape).astype(np.float32)
after += rng.normal(0, 0.4, after.shape).astype(np.float32)
profile = {
    "driver": "GTiff", "height": HEIGHT, "width": WIDTH, "count": 3,
    "dtype": "float32", "crs": "EPSG:32643",
    "transform": from_origin(500000, 2070000, 10, 10), "nodata": -9999.0,
    "compress": "lzw",
}
for name, arr in (("before_synthetic.tif", before), ("after_synthetic.tif", after)):
    with rasterio.open(OUT / name, "w", **profile) as dst:
        dst.write(arr)
print("Created SYNTHETIC test pair (not satellite data):")
print(OUT / "before_synthetic.tif")
print(OUT / "after_synthetic.tif")
print("The added rectangle is an artificial change used only to smoke-test the pipeline and UI.")
