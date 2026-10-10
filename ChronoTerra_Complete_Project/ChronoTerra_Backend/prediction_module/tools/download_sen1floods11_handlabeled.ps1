$ErrorActionPreference = 'Stop'
$Root = 'data/sen1floods11'
$required = Get-Command gsutil -ErrorAction SilentlyContinue
if (-not $required) {
  throw 'gsutil was not found. Install Google Cloud CLI, then open a new PowerShell window. This script deliberately does not install software for you.'
}
New-Item -ItemType Directory -Force -Path "$Root/splits/flood_handlabeled" | Out-Null
New-Item -ItemType Directory -Force -Path "$Root/data/flood_events/HandLabeled/S1Hand" | Out-Null
New-Item -ItemType Directory -Force -Path "$Root/data/flood_events/HandLabeled/LabelHand" | Out-Null
# Downloads only the hand-labeled SAR chips, masks and official split files (not the full ~14 GB dataset).
gsutil -m rsync -r gs://sen1floods11/v1.1/splits/flood_handlabeled "$Root/splits/flood_handlabeled"
gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand "$Root/data/flood_events/HandLabeled/LabelHand"
gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand "$Root/data/flood_events/HandLabeled/S1Hand"
Write-Host 'Download commands completed. Create and inspect the manifest next:'
Write-Host 'python -m tools.make_sen1floods_manifest --root data/sen1floods11 --output data/manifests/sen1floods11.csv'
Write-Host 'Check the split/group counts before training; no model was trained by this download script.'
