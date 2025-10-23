$ErrorActionPreference = "Continue"

# Define required resources
$requiredResources = @{
    "JSON Files" = @(
        "config.json",
        "Codette_Quantum_Harmonic_Baseline_FFT.json",
        "Codette_Integrity_Certificate.json",
        "agischema.json"
    )
    "Model Files" = @(
        "models/*",
        "static/*",
        "data/*",
        "gotchu/*"
    )
    "Package Data" = @(
        "torch-*.dist-info/*",
        "numpy-*.dist-info/*",
        "tokenizers-*.dist-info/*",
        "transformers-*.dist-info/*"
    )
}

# Function to check resource existence in executable
function Test-Resource {
    param (
        [string]$ExePath,
        [string]$ResourcePath
    )
    
    $strings = & strings.exe $ExePath 2>$null
    return $strings | Select-String -Pattern $ResourcePath -Quiet
}

Write-Host "Starting Resource Verification..." -ForegroundColor Green

# Build the executable
Write-Host "`nBuilding Codette with full resources..." -ForegroundColor Yellow
& "K:/Codette the DevUI/Codette/.venv/Scripts/pyinstaller.exe" --clean codette_full.spec

# Verify the build
$exePath = "dist\Codette.exe"
if (!(Test-Path $exePath)) {
    Write-Host "Build failed! Executable not found." -ForegroundColor Red
    exit 1
}

# Check each resource category
foreach ($category in $requiredResources.Keys) {
    Write-Host "`nChecking $category..." -ForegroundColor Yellow
    
    foreach ($resource in $requiredResources[$category]) {
        $exists = Test-Resource -ExePath $exePath -ResourcePath $resource
        $status = if ($exists) { "Found" } else { "Missing" }
        $color = if ($exists) { "Green" } else { "Red" }
        Write-Host "  $resource : $status" -ForegroundColor $color
    }
}

# Create test environment
$testDir = "resource_test"
New-Item -ItemType Directory -Force -Path $testDir | Out-Null
Copy-Item $exePath -Destination $testDir

# Test executable in isolation
Write-Host "`nTesting executable in isolation..." -ForegroundColor Yellow
$process = Start-Process -FilePath "$testDir\Codette.exe" -PassThru
Start-Sleep -Seconds 5

# Check for any new files created
Write-Host "`nChecking for resource extraction..." -ForegroundColor Yellow
$newFiles = Get-ChildItem $testDir -Recurse | Where-Object { $_.CreationTime -gt (Get-Date).AddMinutes(-1) }
if ($newFiles) {
    Write-Host "New files created during execution:" -ForegroundColor Green
    $newFiles | Format-Table Name, CreationTime
} else {
    Write-Host "No new files created during execution." -ForegroundColor Red
}

# Cleanup
Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
Write-Host "`nResource verification complete!" -ForegroundColor Green
