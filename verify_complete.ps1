$ErrorActionPreference = "Continue"

# Define test environment
$testRoot = "complete_test_environment"
$isolatedTest = Join-Path $testRoot "isolated_test"
$resourceTest = Join-Path $testRoot "resource_test"

# Create clean test directories
Write-Host "Creating test environments..." -ForegroundColor Green
Remove-Item -Path $testRoot -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $isolatedTest | Out-Null
New-Item -ItemType Directory -Force -Path $resourceTest | Out-Null

# Copy executable for testing
Write-Host "Preparing test environments..." -ForegroundColor Green
Copy-Item "dist\Codette.exe" -Destination $isolatedTest
Copy-Item "dist\Codette.exe" -Destination $resourceTest

# Resource verification function
function Test-ResourceInExecutable {
    param (
        [string]$ExePath,
        [string]$ResourceName
    )
    
    $content = Get-Content -Path $ExePath -Raw -Encoding Byte
    $textContent = [System.Text.Encoding]::ASCII.GetString($content)
    return $textContent.Contains($ResourceName)
}

# Define critical resources to verify
$criticalResources = @{
    "Configuration Files" = @(
        "config.json",
        "Codette_Quantum_Harmonic_Baseline_FFT.json",
        "Codette_Integrity_Certificate.json",
        "agischema.json"
    )
    "Core Directories" = @(
        "models",
        "static",
        "data",
        "gotchu",
        "cocoons"
    )
    "Neural Network Files" = @(
        "neural_weights",
        "quantum_states"
    )
    "Python Packages" = @(
        "numpy",
        "torch",
        "transformers",
        "vaderSentiment",
        "pymc",
        "arviz"
    )
}

# Verify resources
Write-Host "`nVerifying bundled resources..." -ForegroundColor Green
foreach ($category in $criticalResources.Keys) {
    Write-Host "`n$category:" -ForegroundColor Yellow
    foreach ($resource in $criticalResources[$category]) {
        $found = Test-ResourceInExecutable -ExePath "$isolatedTest\Codette.exe" -ResourceName $resource
        $status = if ($found) { "Found" } else { "Missing" }
        $color = if ($found) { "Green" } else { "Red" }
        Write-Host "  $resource : $status" -ForegroundColor $color
    }
}

# Test executable in isolation
Write-Host "`nTesting executable in isolation..." -ForegroundColor Yellow
$process = Start-Process -FilePath "$isolatedTest\Codette.exe" -PassThru
Start-Sleep -Seconds 5

# Check for process status
if (!$process.HasExited) {
    Write-Host "Executable is running successfully" -ForegroundColor Green
    $process | Select-Object Id, ProcessName, StartTime | Format-List
} else {
    Write-Host "Process exited with code: $($process.ExitCode)" -ForegroundColor Red
}

# Check for created files
Write-Host "`nChecking for runtime-created resources..." -ForegroundColor Yellow
$newFiles = Get-ChildItem $isolatedTest -Recurse | Where-Object { $_.CreationTime -gt (Get-Date).AddMinutes(-1) }
if ($newFiles) {
    Write-Host "New files created during execution:" -ForegroundColor Green
    $newFiles | Format-Table Name, CreationTime, Length
}

# Memory check
$processMemory = Get-Process -Id $process.Id -ErrorAction SilentlyContinue | Select-Object WorkingSet, PM
Write-Host "`nMemory Usage:" -ForegroundColor Yellow
Write-Host "Working Set: $([math]::Round($processMemory.WorkingSet / 1MB, 2)) MB"
Write-Host "Private Memory: $([math]::Round($processMemory.PM / 1MB, 2)) MB"

# Network connection check
Write-Host "`nChecking network connections..." -ForegroundColor Yellow
$connections = Get-NetTCPConnection | Where-Object { $_.OwningProcess -eq $process.Id }
if ($connections) {
    Write-Host "Active network connections found:" -ForegroundColor Red
    $connections | Format-Table LocalAddress, LocalPort, RemoteAddress, RemotePort, State
} else {
    Write-Host "No network connections detected (Good!)" -ForegroundColor Green
}

# DLL dependency check
Write-Host "`nChecking loaded modules..." -ForegroundColor Yellow
$modules = Get-Process -Id $process.Id -Module -ErrorAction SilentlyContinue
Write-Host "Loaded DLLs:"
$modules | Where-Object { $_.ModuleName -notlike "*.exe" } | 
    Select-Object ModuleName, FileName |
    Format-Table -AutoSize

# Clean up
Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
Write-Host "`nVerification complete!" -ForegroundColor Green
