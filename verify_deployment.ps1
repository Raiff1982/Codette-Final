# Codette Package Verification Script
$ErrorActionPreference = "Continue"

# Helper function for colored output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-Output "=== Codette Package Verification ==="
Write-Output "Starting verification process..."

# Test environment setup
$testRoot = "standalone_test_environment"
$testFolders = @(
    "clean_test",      # For testing the exe alone
    "resources_test",  # For testing with minimal required resources
    "network_test"     # For testing network isolation
)

Write-Host "Creating test environments..." -ForegroundColor Green
New-Item -ItemType Directory -Force -Path $testRoot | Out-Null
foreach ($folder in $testFolders) {
    New-Item -ItemType Directory -Force -Path "$testRoot\$folder" | Out-Null
}

# Copy executable to test folders
Write-Host "Copying executable to test environments..." -ForegroundColor Green
foreach ($folder in $testFolders) {
    Copy-Item "dist\Codette.exe" -Destination "$testRoot\$folder" -Force
}

# Network isolation test
Write-Host "`nRunning network isolation test..." -ForegroundColor Green
$netProcess = Start-Process -FilePath "$testRoot\network_test\Codette.exe" -PassThru
Start-Sleep -Seconds 5
$connections = Get-NetTCPConnection | Where-Object { $_.OwningProcess -eq $netProcess.Id }
Write-Host "Network Connections:" -ForegroundColor Yellow
$connections | Format-Table LocalAddress, LocalPort, RemoteAddress, RemotePort, State

# Resource dependency test
Write-Host "`nChecking for resource creation/access..." -ForegroundColor Green
$resProcess = Start-Process -FilePath "$testRoot\resources_test\Codette.exe" -PassThru
Start-Sleep -Seconds 5
Write-Host "New files created:" -ForegroundColor Yellow
Get-ChildItem "$testRoot\resources_test" -Recurse | Where-Object { $_.CreationTime -gt (Get-Date).AddMinutes(-1) } | Select-Object FullName, CreationTime

# Clean environment test
Write-Host "`nTesting in clean environment..." -ForegroundColor Green
$cleanProcess = Start-Process -FilePath "$testRoot\clean_test\Codette.exe" -PassThru
Start-Sleep -Seconds 5

# Process status check
Write-Host "`nChecking process status:" -ForegroundColor Green
$processes = @($netProcess, $resProcess, $cleanProcess)
foreach ($proc in $processes) {
    if (!$proc.HasExited) {
        Write-Host "Process $($proc.Id) is running" -ForegroundColor Green
        $proc | Select-Object Id, ProcessName, StartTime, CPU | Format-List
    } else {
        Write-Host "Process $($proc.Id) has exited with code $($proc.ExitCode)" -ForegroundColor Red
    }
}

# DLL dependency check
Write-Host "`nChecking DLL dependencies:" -ForegroundColor Green
$dllCheck = dumpbin /dependents "$testRoot\clean_test\Codette.exe" 2>&1
if ($LASTEXITCODE -eq 0) {
    $dllCheck
} else {
    Write-Host "Dumpbin not available, using alternative method..." -ForegroundColor Yellow
    $modules = Get-Process -Id $cleanProcess.Id -Module -ErrorAction SilentlyContinue
    $modules | Select-Object ModuleName, FileName | Format-Table
}

# Resource presence check
Write-Host "`nChecking for required resources:" -ForegroundColor Green
$requiredResources = @(
    "config.json",
    "models/*",
    "static/*",
    "data/*"
)

foreach ($resource in $requiredResources) {
    $files = Get-ChildItem -Path "$testRoot\resources_test" -Filter $resource -Recurse -ErrorAction SilentlyContinue
    Write-Host "Resource $resource : $(if($files){'Present'}else{'Missing'})" -ForegroundColor $(if($files){'Green'}else{'Red'})
}

# Cleanup processes
Write-Host "`nCleaning up test processes..." -ForegroundColor Green
foreach ($proc in $processes) {
    if (!$proc.HasExited) {
        $proc | Stop-Process -Force
    }
}

Write-Host "`nVerification complete!" -ForegroundColor Green
