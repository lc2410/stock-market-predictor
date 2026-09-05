param (
    [switch]$sync
)

if ($sync) {
    Write-Host "🔄 Syncing latest Oracle Wallet and Secrets from production server..." -ForegroundColor Cyan
    
    # 1. Download and extract wallet
    scp -i "$env:USERPROFILE\.ssh\github_actions" -o StrictHostKeyChecking=no ubuntu@150.136.47.42:/home/ubuntu/wallet.zip wallet.zip
    Expand-Archive -Force -Path wallet.zip -DestinationPath wallet
    Remove-Item wallet.zip
    
    # Update Oracle configuration to point to the absolute local directory
    $walletPath = (Get-Item -Path ".\wallet").FullName -replace '\\', '\\\\'
    $sqlnetPath = ".\wallet\sqlnet.ora"
    $content = Get-Content $sqlnetPath
    $content = $content -replace '\?/network/admin', $walletPath
    $content = $content -replace '/home/ubuntu/wallet', $walletPath
    Set-Content -Path $sqlnetPath -Value $content
    
    # 2. Extract database credentials from production systemd service
    Write-Host "Fetching environment variables..." -ForegroundColor Cyan
    $remoteConfig = ssh -i "$env:USERPROFILE\.ssh\github_actions" -o StrictHostKeyChecking=no ubuntu@150.136.47.42 "cat /etc/systemd/system/marketlens.service"
    
    $envContent = @()
    foreach ($line in $remoteConfig) {
        if ($line -match '^Environment="DB_(.*?)"$') {
            $kv = $line -replace '^Environment="', '' -replace '"$', ''
            # Split into key and value, then format with single quotes for safety
            $split = $kv -split '=', 2
            if ($split.Length -eq 2) {
                $envContent += "$($split[0])='$($split[1])'"
            }
        }
    }
    
    # Add local TNS_ADMIN path
    $envContent += "TNS_ADMIN='$((Get-Item -Path ".\wallet").FullName)'"
    Set-Content -Path .env -Value $envContent
    
    Write-Host "✅ Wallet synced, secrets downloaded, and configured for local development!" -ForegroundColor Green
    Write-Host "------------------------------------------------------"
}

# Export all variables from .env
if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^(?<name>[^=]+)=(?<value>.*)$') {
            $name = $Matches.name.Trim()
            $value = $Matches.value.Trim(" '`"")
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
} else {
    Write-Host "❌ Warning: .env file not found! Try running with -sync" -ForegroundColor Red
}

# Start the Flask API
python app.py
