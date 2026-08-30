#!/bin/bash
# update_agent.sh — Wrapper script for the launchd background agent.
# Activates the virtualenv and runs the database update function.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV_DIR="$PROJECT_DIR/stock-market-predictor-env"
LOG_FILE="$SCRIPT_DIR/update_agent.log"

echo "========================================" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Agent triggered." >> "$LOG_FILE"

# Activate virtualenv
source "$VENV_DIR/bin/activate"

# Run the update from the project root so relative imports resolve
cd "$PROJECT_DIR"
python -c "from backend.database.scripts.update_db import update_database; update_database()" >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished with exit code $EXIT_CODE." >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

exit $EXIT_CODE
