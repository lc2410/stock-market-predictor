#!/bin/bash

# Sync latest wallet and secrets from production if requested
if [ "$1" == "--sync" ]; then
    echo "🔄 Syncing latest Oracle Wallet and Secrets from production server..."
    
    # 1. Download and extract wallet
    scp -i ~/.ssh/github_actions -o StrictHostKeyChecking=no ubuntu@150.136.47.42:/home/ubuntu/wallet.zip wallet.zip
    unzip -o wallet.zip -d wallet
    rm wallet.zip
    
    # Update Oracle configuration to point to the absolute local directory
    sed -i '' "s|?/network/admin|$(pwd)/wallet|g" wallet/sqlnet.ora
    sed -i '' "s|/home/ubuntu/wallet|$(pwd)/wallet|g" wallet/sqlnet.ora
    
    # 2. Extract database credentials from production systemd service
    echo "Fetching environment variables..."
    ssh -i ~/.ssh/github_actions -o StrictHostKeyChecking=no ubuntu@150.136.47.42 "cat /etc/systemd/system/marketlens.service" | grep 'Environment="DB_' | sed 's/Environment="//g' | sed 's/"//g' > .env.temp
    
    # Wrap values in single quotes to prevent bash parsing errors and format .env
    > .env
    while IFS='=' read -r key value; do
        if [ -n "$key" ]; then
            echo "${key}='${value}'" >> .env
        fi
    done < .env.temp
    rm .env.temp
    
    # Add local TNS_ADMIN path
    echo "TNS_ADMIN='$(pwd)/wallet'" >> .env
    
    echo "✅ Wallet synced, secrets downloaded, and configured for local development!"
    echo "------------------------------------------------------"
fi

# Export all variables from .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "❌ Warning: .env file not found! Try running with --sync"
fi

# Start the Flask API
python3 app.py
