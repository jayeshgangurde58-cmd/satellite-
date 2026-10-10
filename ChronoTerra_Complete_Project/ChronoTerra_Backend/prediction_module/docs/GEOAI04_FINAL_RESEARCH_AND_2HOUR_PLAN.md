# GEOAI-04 — Final Research Audit and Two-Hour Prototype Plan

**Prepared:** 9 October 2026  
**Scope:** FUSION 2026 GEOAI-04 — Bi-Temporal Satellite Change Detection for Rapid Urban Encroachment & Disaster Impact  
**Status:** Evidence-led recommendation; candidate data routes documented, but no real satellite pair has yet been downloaded and validated in this environment.

## Executive decision

**NARROW GO.** Continue with a quality-gated, auditable change-analysis prototype. Do not claim a new change-detection architecture, a solution to domain generalization, or validated disaster-damage severity. The immediate goal is to make an honest demo that (1) accepts suitable georeferenced before/after rasters, (2) runs a CPU-first baseline, (3) exposes candidate changes, (4) exports correct geospatial features, and (5) communicates input and output limitations.

Two separate data needs must not be conflated:

- **Pixel/feature performance evaluation:** use a labelled dataset such as OSCD, or SpaceNet 7 building labels if individual files and terms are verified.
- **True GIS proof:** use an actual georeferenced before/after GeoTIFF pair and verify CRS, affine transform, bounds, resolution, nodata, alignment and output locations. OSCD's TorchGeo wrapper is a `NonGeoDataset`; that fact alone does not prove every original TIFF lacks georeferencing, so inspect the original files before deciding.

## 1. Official problem requirements

Source: the attached FUSION 2026 GEOAI problem statement, section GEOAI-04. It calls for co-registered bi-temporal satellite/aerial imagery, change mapping, false-alarm filtering (including seasonal vegetation and sun-angle differences), attributed polygons (area, magnitude, centroid), GIS-ready GeoJSON/GeoPackage, severity tiers and an interactive before/after split-view.

| Requirement | Minimum acceptable prototype | Acceptance evidence |
|---|---|---|
| Input pair | Two georeferenced, same-footprint rasters with known bands | Metadata report; reject unsupported grid mismatch |
| Change map | Classical change-vector baseline, optional histogram matching | Mask and threshold saved; baseline settings recorded |
| False-alarm handling | Valid-pixel mask, cloud/nodata handling when supplied, morphological filtering | Per-scene metrics/visual review; no blanket robustness claims |
| Vectorization | Polygonize binary mask with original affine transform | Geometry validity checks and overlay on source imagery |
| Attributes | `area_m2`, `change_magnitude`, centroid lon/lat, change-priority tier | Independently verify area/centroid coordinates |
| GIS exports | GeoJSON, optionally GeoPackage; EPSG:4326 vectors | Reopen exports and inspect geometries/CRS |
| Swipe interface | Before/after overlay, polygon overlay, downloadable results | Upload-to-download smoke test |
| Severity | Call tiers *change priority* unless damage labels support more | UI, documentation and pitch use defensible terminology |

GeoJSON (RFC 7946) coordinates are WGS 84 longitude/latitude decimal degrees; area must be calculated in a suitable projected CRS rather than directly in angular degrees. Sources: https://www.rfc-editor.org/rfc/rfc7946.html and https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.area.html.

## 2. Real-world need and what already exists

Copernicus EMS explains that automatic extraction can perform well when multi-temporal images share sensor/resolution and homogeneous radiometric response, but is rarely used in rapid mapping when pre/post homogeneity cannot be guaranteed. Its guidance also describes limits of image-based structural damage inference, including view angle and spatial/radiometric resolution. This validates the operational problem, not a claim that current systems lack change mapping.

Source: https://mapping.emergency.copernicus.eu/about/rapid-mapping-manual/detection-methods-damage-assessment/

Other real operational references include:
- Copernicus EMS rapid mapping: https://mapping.emergency.copernicus.eu/about/
- USGS/MTBS burn-severity products: https://www.mtbs.gov/
- NASA disaster-practitioner resources: https://appliedsciences.nasa.gov/what-we-do/disasters/practitioner-resources

Project positioning: an auditable analyst-support tool for candidate change triage, not a replacement for authoritative emergency-mapping services or ground inspection.

## 3. Literature audit: don't claim generic novelty

| Work | Established prior art | Design consequence |
|---|---|---|
| Daudt et al., IGARSS 2018, OSCD | Multispectral urban change detection with Sentinel-2 | Multiband urban change detection is not new. https://doi.org/10.1109/IGARSS.2018.8518015 |
| Chen & Shi, 2020, LEVIR-CD | High-resolution building-change benchmark | Building growth/removal with labelled change has a mature benchmark. https://justchenhao.github.io/LEVIR/ |
| AnyChange, NeurIPS 2024 | Training-free zero-shot change detection using SAM latent matching | “We use a foundation model without training” alone is not novelty. https://proceedings.neurips.cc/paper_files/paper/2024/hash/9415416201aa201902d1743c7e65787b-Abstract-Conference.html |
| Cross-domain review, IEEE JSTARS 2024 | Reviews cross-domain approaches | Domain shift/adaptation is already established literature. https://doi.org/10.1109/JSTARS.2024.3416183 |
| ICCD / SemiSiamCD, Results in Engineering 2026 | Indian-city building-change data and semi-supervised cross-city approach | Do not claim Indian-city generalization/uncertainty-aware CD is unstudied. https://doi.org/10.1016/j.rineng.2025.108880 |
| BRIGHT, ESSD 2025 | Global multimodal optical/SAR building-damage dataset | Optical/SAR damage assessment already has benchmark work; too much integration risk for the first MVP. https://doi.org/10.5194/essd-17-6217-2025 |
| CrossCD-Eval, Scientific Reports, published 3 Oct 2026 | Cross-domain and few-shot evaluation of recent methods; domain-aware feature normalization | Current evidence shows cross-domain degradation remains, but it is not a new research topic. https://doi.org/10.1038/s41598-026-70914-7 |
| Systematic review, Remote Sensing 2026 | Review of 144 studies; highlights benchmark protocol and research-to-deployment gaps | Supports measuring deployability/benchmark quality, not claiming our integration is novel. https://doi.org/10.3390/rs18152573 |

### Defensible candidate contribution

**Quality-gated change intelligence:** check the pair's metadata and valid overlap; expose a residual-shift diagnostic as a warning rather than auto-warping based on it; mask invalid/cloud pixels; compare baseline with and without histogram matching; report precision/recall/F1/IoU and conventional FPR; export auditable GIS features; flag uncertain/unsuitable inputs.

This is currently an engineering hypothesis, not a verified scientific gap. It needs evidence that the quality checks prevent bad outputs or that an enhancement improves measured results. Do not claim universal false-positive suppression or domain generalization.

## 4. Dataset and access strategy

### A. SpaceNet 7 — best labelled urban-building-change candidate

Official source: https://www.spacenet.ai/sn7-challenge/  
Paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Van_Etten_The_Multi-Temporal_Urban_Development_SpaceNet_Dataset_CVPR_2021_paper.pdf

Provider docs report 101 AOIs, 2,389 observations, ~11 million building annotations and mean GSD ~4 m. It provides monthly Planet imagery and building-footprint labels and states CC BY-SA 4.0; an AWS account is required per the official challenge page. The full tarball is large; retrieve a single tile's two monthly TIFFs and two matching building-footprint GeoJSONs if possible. A published example references tile `L15-0357E-1223N_1429_3296_13`, January 2018 and January 2020. Source for that example: https://opengeoai.org/examples/rastervision/semantic_segmentation/.

**Mandatory before use:** verify exact image/label paths, access/request payment policy, file sizes, per-file CRS/affine transform/bounds/dimensions/bands/masks, label difference, and visible change. Monthly mosaics are not individual acquisition timestamps. Labels are building footprints, not disaster-damage severity. The public challenge page's copied sample command and older secondary sources have inconsistent path spelling; list the bucket and verify exact keys before download.

### B. Microsoft Planetary Computer — best real geospatial demonstration route

STAC collection: https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a  
Dataset overview: https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a  
SAS token/access docs: https://planetarycomputer.microsoft.com/docs/concepts/sas/  
Copernicus terms: https://cds.climate.copernicus.eu/licences/ec-sentinel

The STAC collection describes a global Sentinel-2 L2A archive in COGs and 10/20/60 m GSD. Blue B02, green B03, red B04 and NIR B08 are 10 m; red-edge/SWIR bands are 20 m and some atmospheric bands are 60 m. Public catalog searching does not need a user subscription key; asset access may require signed SAS URLs, supported by `planetary-computer` Python SDK.

Preferred demonstration scope: use same MGRS tile, same four native 10 m bands, similar season/date windows, low item-level cloud cover, SCL/valid-data masks, item IDs and provenance in a saved metadata JSON. `eo:cloud_cover` is scene-level and not a guarantee that the AOI is clear. A shared grid/transform is not proof of content registration.

**Radiometry:** Sentinel-2 Processing Baseline 04.00 introduced a DN shift after 24 January 2022. Earth Engine's harmonized collection documentation explains the adjustment. Planetary Computer's Sentinel-2 L2A COGs are not automatically identical to Earth Engine's `COPERNICUS/S2_SR_HARMONIZED`. Record each item’s processing baseline and scale/offset metadata; do not call raw DN magnitudes calibrated reflectance. If baseline/scaling cannot be normalized within the time budget, label the method raw-DN change magnitude and avoid radiometric conclusions.

Sources: https://sentiwiki.copernicus.eu/web/s2-products and https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED.

### C. OSCD — labelled pixel metric benchmark

Official source: https://rcdaudt.github.io/oscd/  
Original paper: https://doi.org/10.1109/IGARSS.2018.8518015  
TorchGeo API: https://docs.torchgeo.org/en/stable/api/datasets/oscd.html

The official page reports 24 registered 13-band Sentinel-2 pairs (2015–2018), 14 train and 10 test pairs, with pixel labels for all pairs; native band resolutions vary 10/20/60 m. Use OSCD for labelled pixel-level change evaluation, not to infer building-level damage or to prove correct map coordinates without inspecting the actual original files' metadata. A `NonGeoDataset` designation in a library wrapper is not proof that every raw TIFF lacks georeferencing; inspect file metadata directly.

### D. OpenAerialMap — discovery fallback only

Catalog: https://imagery.hotosm.org/  
Docs: https://docs.imagery.hotosm.org/usage/using-imagery/  
License terms: https://openaerialmap.org/legal/  
STAC schema: https://github.com/hotosm/openaerialmap/blob/main/backend/stactools-hotosm/stac-extension/README.md

It offers imagery, metadata and item-level licensing, but there is no verified matched pre/post pair yet. The general legal page describes CC BY 4.0 licensing for OIN submissions, while the current item schema allows other Creative Commons license values; inspect each item. Do not treat it as a ready-to-use pair until two records match in area, resolution, date and license.

### E. Camp Fire / MTBS — thematic fallback

USGS before/after page: https://www.usgs.gov/media/before-after/camp-fire-california-2018  
MTBS direct download: https://burnseverity.cr.usgs.gov/direct-download?product=MTBS  
Product descriptions: https://www.mtbs.gov/product-descriptions

USGS documents a public-domain visual comparison with 7 Oct 2018 and 8 Nov 2018 imagery. The event-level MTBS package offers reflectance and burn-severity products, masks and perimeter vectors, but the imagery acquisition dates in the package must come from its own metadata; do not assume they are identical to the web-comparison dates. At 30 m, this is suitable for broad burn/land-cover change and GIS checks, not individual-building damage. Burn-severity classes are not independent ground truth for a spectral-difference method that uses similar spectra.

## 5. Prototype code audit / implementation gotchas

1. `pipeline/change.py`: valid-pixel-aware histogram matching is implemented, but it requires continued testing on real raster masks and non-finite data.
2. Evaluation reports both the standard FPR `FP/(FP+TN)` and false-positive fraction among all valid pixels. Do not conflate them.
3. `pipeline/geo_io.py`: equal CRS/dimensions/affine transform is a grid compatibility check, not a proof of content-level co-registration.
4. `pipeline/vectorize.py`: area is computed in an estimated UTM CRS; verify that this is appropriate for the scene extent, especially for large/multi-zone geometries. Centroids should be calculated in a suitable projected CRS and then converted to WGS84.
5. `severity.py`: heuristic tiers represent change priority only. Do not use Low/Medium/High/Critical as physical damage grades without suitable damage labels.
6. `tools/fetch_sentinel_pair.py`: real STAC selection and asset download have not succeeded in this research runtime. Check whether a query returns two items in the same MGRS tile; inspect both outputs and all masks. The downloader picks a pair using cloud/platform/baseline scores but a scoring penalty is not a guarantee of matching radiometry. Prefer rejecting incompatible baseline/processing if the selected pair cannot be normalized.
7. The web UI uses CDN Leaflet assets and OpenStreetMap tiles. The UI needs internet for those resources, although the raster analysis is local. If plugin loading fails, show a clear fallback and retain artifact downloads.
8. The synthetic sample is solely a pipeline/UI smoke test, not satellite evidence.

## 6. Evaluation protocol

Metrics per scene and aggregate:
- Precision = TP/(TP+FP)
- Recall = TP/(TP+FN)
- F1 = 2PR/(P+R)
- IoU = TP/(TP+FP+FN)
- Conventional FPR = FP/(FP+TN)
- False-positive fraction of valid pixels = FP/number-of-valid-pixels (a different measure)
- TP/FP/FN/TN, valid-pixel count, runtime

Report macro per-scene metrics and micro metrics pooled from counts. Never headline pixel accuracy alone. Preserve original train/development and test split; tune parameters on train/development only. For a two-hour prototype, test pair-loading and metrics first; do not claim statistically generalizable robustness from 10 scenes or one real pair.

Required ablation: no histogram matching vs valid-pixel histogram matching, using exactly the same scenes and baseline parameters otherwise. Further ablations should only be added if time allows.

## 7. Immediate two-hour plan

**0–15 min:** install the starter requirements in Python 3.11; run `python -m pytest -q`; if tests fail, save the exact traceback and fix only P0 issues.

**15–35 min:** generate synthetic sample pair and run the full UI upload → analyze → polygon → download path. Label it SYNTHETIC in the UI and presentation.

**35–70 min:** try the Planetary Computer fetch command from README for a small Pune urban AOI and same-season time windows. If it fails, save the error and do not invent an item ID. If it succeeds, inspect `pair_metadata.json`, before/after previews and residual shift diagnostic.

**70–95 min:** run the actual real pair through the API; confirm GeoJSON reopens, geometry is inside raster bounds and area/centroid values are plausible. Do not report accuracy without ground-truth labels.

**95–110 min:** if OSCD archives are available, run the one-scene test; if not, do not waste all remaining time on a large download. Prepare one honest slide documenting data source, verified metadata, baseline and its limitations.

**110–120 min:** rehearse a 90-second demo and show one artifact download. Prepare screenshot/video fallback if the network or map CDN fails.

## 8. Acceptance checklist before saying “prototype works”

- [ ] `python -m pytest -q` passes on the target environment; record Python/OS/dependency versions.
- [ ] Synthetic upload-to-download test works and is labelled synthetic.
- [ ] Real pair source, license, dates and IDs are saved.
- [ ] Actual raster metadata are inspected; two files share the expected grid.
- [ ] Visual/content alignment reviewed; shift diagnostic is treated as indicative only.
- [ ] Nodata/cloud mask and common valid overlap checked.
- [ ] Change mask and polygons appear in expected geography.
- [ ] EPSG:4326 export uses longitude/latitude coordinates per RFC 7946.
- [ ] Metric area and centroid tested independently.
- [ ] No damage-severity or accuracy claim is made without appropriate labels.
- [ ] Benchmark results are not invented; real-image no-label demos are described as visual inspections, not measured performance.

## 9. Recommended decision

Use the starter's CPU-first classical baseline and FastAPI/Leaflet prototype. Use **OSCD for labelled pixel-level evaluation**, **Planetary Computer for a small real georeferenced demonstration if the live STAC query works**, and **SpaceNet 7 for high-resolution building-change evaluation if per-object retrieval/labels/terms are verified**. These are complementary, not interchangeable datasets.

No further broad literature search is a prerequisite to the basic demo. Remaining high-value work is empirical: download/inspect a real pair, run baseline metrics, test the nodata/histogram correction, and validate GIS exports. If the real pair cannot be obtained before the round, demo the synthetic path transparently and describe the real-data route as not yet validated.

## Primary references

- Official GEOAI-04 PDF supplied in the conversation.
- OSCD: https://rcdaudt.github.io/oscd/
- OSCD paper: https://doi.org/10.1109/IGARSS.2018.8518015
- LEVIR-CD: https://justchenhao.github.io/LEVIR/
- SpaceNet 7: https://www.spacenet.ai/sn7-challenge/
- SpaceNet 7 paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Van_Etten_The_Multi-Temporal_Urban_Development_SpaceNet_Dataset_CVPR_2021_paper.pdf
- Planetary Computer Sentinel-2 STAC: https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a
- Planetary Computer SAS: https://planetarycomputer.microsoft.com/docs/concepts/sas/
- Copernicus terms: https://cds.climate.copernicus.eu/licences/ec-sentinel
- Sentinel-2 product guide: https://sentiwiki.copernicus.eu/web/s2-products
- Harmonized Sentinel-2 L2A: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
- AnyChange: https://proceedings.neurips.cc/paper_files/paper/2024/hash/9415416201aa201902d1743c7e65787b-Abstract-Conference.html
- CrossCD-Eval: https://doi.org/10.1038/s41598-026-70914-7
- ICCD: https://doi.org/10.1016/j.rineng.2025.108880
- BRIGHT: https://doi.org/10.5194/essd-17-6217-2025
- 2026 systematic review: https://doi.org/10.3390/rs18152573
- Copernicus EMS detection/damage: https://mapping.emergency.copernicus.eu/about/rapid-mapping-manual/detection-methods-damage-assessment/
- Rasterio features: https://rasterio.readthedocs.io/en/stable/api/rasterio.features.html
- GeoPandas area: https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.area.html
- GeoJSON RFC 7946: https://www.rfc-editor.org/rfc/rfc7946.html
- OpenAerialMap terms: https://openaerialmap.org/legal/
- MTBS direct data: https://burnseverity.cr.usgs.gov/direct-download?product=MTBS

## 10. Local verification performed for this pack

The available container was Python 3.13 (not the target Windows/Python 3.11 environment).

- `python -m pytest -q`: **17 passed, 4 PyProj deprecation warnings**.
- `python -m compileall -q pipeline evaluation api tools`: completed successfully.
- `python tools/fetch_sentinel_pair.py --help`: completed successfully; this verifies CLI wiring, not remote data access.
- `python -m uvicorn api.app:app --host 127.0.0.1 --port 8765`: `/` returned HTTP 200 and `/api/health` returned `{"status":"ok","mode":"local-cpu-baseline",...}`.
- Real HTTP end-to-end using the synthetic GeoTIFF pair: `POST /api/analyze` returned 200, found 1 candidate polygon in the synthetic scene, and GeoJSON, GeoPackage, and mask download endpoints each returned HTTP 200. This proves local API/file flow only; the sample was synthetically generated, not satellite imagery.
- The live Planetary Computer query/download was **not verified** in this environment. No real STAC item IDs, real-pair sizes, real-image metric scores, or real-world accuracy values are claimed.

Warnings: four PyProj deprecation warnings arose from centroid/projection code. They did not fail the tests, but should be cleaned up when time allows. Python 3.11 on a Windows laptop remains a required verification.
