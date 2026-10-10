$ErrorActionPreference = 'Stop'
$downloadDir = Join-Path (Get-Location) 'data\downloads'
$extractDir = Join-Path (Get-Location) 'data\oscd'
New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

$archives = @(
    @{ Name = 'oscd_images.zip'; Url = 'https://partage.imt.fr/index.php/s/gKRaWgRnLMfwMGo/download' },
    @{ Name = 'oscd_train_labels.zip'; Url = 'https://partage.mines-telecom.fr/index.php/s/2D6n03k58ygBSpu/download' },
    @{ Name = 'oscd_test_labels.zip'; Url = 'https://partage.imt.fr/index.php/s/gpStKn4Mpgfnr63/download' }
)

foreach ($item in $archives) {
    $target = Join-Path $downloadDir $item.Name
    if (-not (Test-Path $target) -or (Get-Item $target).Length -eq 0) {
        Write-Host "Downloading $($item.Name) from the maintainer-linked mirror..."
        & curl.exe -L --fail --retry 3 --retry-delay 2 --output $target $item.Url
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $target) -or (Get-Item $target).Length -eq 0) {
            throw "Download failed for $($item.Name). Open https://rcdaudt.github.io/oscd/ and use its current download links; do not use an empty or HTML error file."
        }
    } else {
        Write-Host "Using existing archive $target"
    }
    # Validate that it is a ZIP before extraction.
    $sig = [System.IO.File]::ReadAllBytes($target)[0..3]
    if ($sig[0] -ne 0x50 -or $sig[1] -ne 0x4B) {
        throw "$target does not appear to be a ZIP. Check the downloaded content in a browser."
    }
    Expand-Archive -LiteralPath $target -DestinationPath $extractDir -Force
}

Write-Host "Extraction complete. Verify these directories exist:"
Get-ChildItem -LiteralPath $extractDir -Directory | Select-Object -ExpandProperty Name
Write-Host "Next: python -m models.train_oscd --root data/oscd --output outputs/models/oscd_siamese_v2 --epochs 30 --steps-per-epoch 100"
