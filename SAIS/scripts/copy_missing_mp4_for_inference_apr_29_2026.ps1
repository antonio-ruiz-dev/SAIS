param(
    [string]$SourceDir = (Join-Path $PSScriptRoot "..\original-videos"),
    [string]$TrainingDir = (Join-Path $PSScriptRoot "..\videos-used-for-training-apr-29-2026"),
    [string]$InferenceDir = (Join-Path $PSScriptRoot "..\videos-used-for-inference-apr-29-2026")
)

$ErrorActionPreference = "Stop"

$SourceDir = (Resolve-Path $SourceDir).Path

if (-not (Test-Path -LiteralPath $TrainingDir)) {
    throw "Training directory not found: $TrainingDir"
}
$TrainingDir = (Resolve-Path $TrainingDir).Path

if (-not (Test-Path -LiteralPath $InferenceDir)) {
    New-Item -ItemType Directory -Path $InferenceDir -Force | Out-Null
}
$InferenceDir = (Resolve-Path $InferenceDir).Path

Write-Host "Source:    $SourceDir"
Write-Host "Training:  $TrainingDir"
Write-Host "Inference: $InferenceDir"

# Build a case-insensitive set of .mp4 file names present in the training folder.
$trainingNames = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
Get-ChildItem -Path $TrainingDir -Filter *.mp4 -File -Recurse | ForEach-Object {
    [void]$trainingNames.Add($_.Name)
}

$sourceFiles = Get-ChildItem -Path $SourceDir -Filter *.mp4 -File -Recurse
$copiedCount = 0
$skippedCount = 0

foreach ($file in $sourceFiles) {
    if ($trainingNames.Contains($file.Name)) {
        $skippedCount++
        continue
    }

    # Preserve relative folder structure under the inference directory.
    $relativePath = $file.FullName.Substring($SourceDir.Length) -replace '^[\\/]+'
    $destinationPath = Join-Path $InferenceDir $relativePath
    $destinationFolder = Split-Path -Path $destinationPath -Parent

    if (-not (Test-Path -LiteralPath $destinationFolder)) {
        New-Item -ItemType Directory -Path $destinationFolder -Force | Out-Null
    }

    Copy-Item -Path $file.FullName -Destination $destinationPath -Force
    $copiedCount++
}

Write-Host "Done."
Write-Host "Total source .mp4 files: $($sourceFiles.Count)"
Write-Host "Skipped (already in training by filename): $skippedCount"
Write-Host "Copied to inference: $copiedCount"
