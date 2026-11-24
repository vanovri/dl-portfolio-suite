# Sequential CUDA training runner for GTX 1650 Ti (4 GB VRAM, zero contention)
param(
  [string[]]$Only = @()
)

$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$pyDL = Join-Path $root ".venv-dl\Scripts\python.exe"

$projects = @(
  "01_eye_disease_gradcam",
  "02_cervical_cancer_segmentation",
  "03_corrosion_unet_segmentation",
  "04_leaf_disease_transfer_learning",
  "05_indo_car_classifier_marketplace",
  "06_yolov8_detection_segmentation",
  "07_wesad_stress_bilstm",
  "08_indobert_sentiment_nlp",
  "09_landslide_ai_sdss",
  "10_icecube_neutrino_3d",
  "11_multimodal_image_audio_fusion",
  "12_hybrid_quantum_dl",
  "13_humanities_nusantara_dl"
)

if ($Only.Count -gt 0) {
  $projects = $projects | Where-Object { $p = $_; $Only | Where-Object { $p -like "*$_*" } }
}

$failed = @()
foreach ($dir in $projects) {
  $projPath = Join-Path $root $dir
  $script = Join-Path $projPath "train.py"
  if (-not (Test-Path $script)) {
    Write-Output "[SKIP] $dir (missing train.py)"
    $failed += $dir
    continue
  }
  Write-Output "============================================================"
  Write-Output "[RUN]  $dir"
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  Push-Location $projPath
  $env:PYTHONPATH = "$root;$projPath"
  & $pyDL train.py
  $rc = $LASTEXITCODE
  Pop-Location
  $sw.Stop()
  $sec = [math]::Round($sw.Elapsed.TotalSeconds, 1)
  if ($rc -ne 0) {
    Write-Output "[FAIL] $dir | rc=$rc | ${sec}s"
    $failed += $dir
  } else {
    Write-Output "[PASS] $dir | rc=0 | ${sec}s"
  }
}

Write-Output "============================================================"
Write-Output ("SUMMARY: total={0} | failed={1}" -f $projects.Count, $failed.Count)
if ($failed.Count -gt 0) {
  Write-Output ("FAILED_LIST: " + ($failed -join ", "))
  exit 1
}
