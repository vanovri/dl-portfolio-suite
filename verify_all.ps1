# Unified verification sweep for all 13 Deep Learning Portfolio Suite projects
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pyDL = Join-Path $root ".venv-dl\Scripts\python.exe"
$env:CUDA_VISIBLE_DEVICES = ""

$jobs = @(
  @{ id="01"; dir="01_eye_disease_gradcam";            req="reports\metrics.json" },
  @{ id="02"; dir="02_cervical_cancer_segmentation";   req="reports\metrics.json" },
  @{ id="03"; dir="03_corrosion_unet_segmentation";    req="reports\metrics.json" },
  @{ id="04"; dir="04_leaf_disease_transfer_learning"; req="reports\metrics.json" },
  @{ id="05"; dir="05_indo_car_classifier_marketplace";req="reports\metrics.json" },
  @{ id="06"; dir="06_yolov8_detection_segmentation";  req="reports\metrics.json" },
  @{ id="07"; dir="07_wesad_stress_bilstm";            req="reports\metrics.json" },
  @{ id="08"; dir="08_indobert_sentiment_nlp";         req="reports\metrics.json" },
  @{ id="09"; dir="09_landslide_ai_sdss";              req="reports\metrics.json" },
  @{ id="10"; dir="10_icecube_neutrino_3d";            req="reports\metrics.json" },
  @{ id="11"; dir="11_multimodal_image_audio_fusion";  req="reports\metrics.json" },
  @{ id="12"; dir="12_hybrid_quantum_dl";              req="reports\metrics.json" },
  @{ id="13"; dir="13_humanities_nusantara_dl";        req="reports\metrics.json" }
)

$allPass = $true
$totalPassed = 0
foreach ($j in $jobs) {
  $p = Join-Path $root $j.dir
  $reqOk = Test-Path (Join-Path $p $j.req)
  Push-Location $p
  $env:PYTHONPATH = "$root;$p"
  $out = & $pyDL -m pytest -o addopts="" -W ignore tests 2>&1
  $rc = $LASTEXITCODE
  Pop-Location
  $lineObj = $out | Select-String -Pattern "\d+ passed|\d+ failed|\d+ error" | Select-Object -Last 1
  $summary = if ($lineObj) { $lineObj.ToString().Trim() } else { ($out | Select-Object -Last 2) -join " | " }
  if ($summary -match "(\d+) passed") { $totalPassed += [int]$Matches[1] }
  if ($rc -ne 0 -or -not $reqOk) { $allPass = $false }
  Write-Output ("[{0}] {1,-36} | rc={2} | req={3} | {4}" -f $j.id, $j.dir, $rc, $reqOk, $summary)
}
Write-Output "=== ALL_PASS: $allPass | TOTAL_PASSED: $totalPassed ==="
if (-not $allPass) { exit 1 }
