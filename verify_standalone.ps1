# Create a test directory
$testDir = "standalone_test"
New-Item -ItemType Directory -Force -Path $testDir

# Copy only the executable
Copy-Item "dist\Codette.exe" -Destination $testDir

# Check file size and hash
Write-Host "Executable Details:"
Get-Item "$testDir\Codette.exe" | Select-Object Length,CreationTime,LastWriteTime | Format-List

# List all files in the directory
Write-Host "`nContents of test directory:"
Get-ChildItem $testDir -Recurse | Select-Object FullName

# Check if the executable runs (don't wait for it to complete)
Write-Host "`nAttempting to start the executable..."
Start-Process -FilePath "$testDir\Codette.exe" -NoNewWindow

# Wait a moment to check for immediate crashes
Start-Sleep -Seconds 5

# Check for any new network connections created by the process
Write-Host "`nChecking network connections..."
Get-NetTCPConnection | Where-Object { $_.State -eq "Established" } | 
    Select-Object LocalAddress,LocalPort,RemoteAddress,RemotePort,State,OwningProcess |
    Format-Table

# Check for any new files created
Write-Host "`nChecking for new files created..."
Get-ChildItem $testDir -Recurse | Where-Object { $_.CreationTime -gt (Get-Date).AddMinutes(-1) }
