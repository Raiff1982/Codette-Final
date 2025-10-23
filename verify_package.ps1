$ErrorActionPreference = "Continue"

Write-Host "=== Codette Package Verification ==="
Write-Host "Starting verification process..."

# Test executable
Write-Host "`nTesting executable..."
try {
    $result = & ".\dist\test_codette_exe.exe" "--test"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Executable test passed" -ForegroundColor Green
    } else {
        Write-Host "✗ Executable test failed" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ Error running executable: $_" -ForegroundColor Red
}

# Test fallback models
Write-Host "`nTesting fallback models..."
try {
    $result = & "python" "test_fallback_models.py"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Fallback models test passed" -ForegroundColor Green
    } else {
        Write-Host "✗ Fallback models test failed" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ Error testing fallback models: $_" -ForegroundColor Red
}

# Verify required files
Write-Host "`nVerifying required files..."
$requiredFiles = @(
    ".\dist\test_codette_exe.exe",
    ".\models\fallback\__init__.py",
    ".\models\fallback\model_config.json"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "✓ Found $file" -ForegroundColor Green
    } else {
        Write-Host "✗ Missing $file" -ForegroundColor Red
    }
}

# Check model downloads
Write-Host "`nVerifying model downloads..."

# Use the verify_models.py script
$pythonScript = Join-Path $PSScriptRoot "verify_models.py"

try {
    $env:PYTHONPATH = Join-Path $PSScriptRoot "dist"
    $result = & python $pythonScript
    if ($LASTEXITCODE -eq 0 -and $result -contains "SUCCESS") {
        Write-Host "✓ Model download check passed" -ForegroundColor Green
    } else {
        Write-Host "✗ Model download check failed: $result" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ Model download check failed: $_" -ForegroundColor Red
}

# Cleanup
if (Test-Path $pythonScript) {
    Remove-Item $pythonScript -Force
}

# Run Python verification script
Write-Host "`nVerifying model downloads..."
$pythonPath = Join-Path $PSScriptRoot "dist"
$env:PYTHONPATH = $pythonPath
$verifyScript = Join-Path $PSScriptRoot "verify_models.py"

if (-not (Test-Path $verifyScript)) {
    Write-Host "✗ Verification script not found: $verifyScript" -ForegroundColor Red
} else {
    try {
        $result = & python $verifyScript 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Model verification passed" -ForegroundColor Green
        } else {
            Write-Host "✗ Model verification failed" -ForegroundColor Red
            Write-Host $result
        }
    } catch {
        Write-Host "✗ Error running verification: $_" -ForegroundColor Red
    }
}

# Summary
Write-Host "`nVerification complete!"
Write-Host "========================================="
