@echo off
REM update_agent.bat — Wrapper script for the Windows Task Scheduler background agent.
REM Activates the virtualenv and runs the database update function.

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%..\..\.."
set "VENV_DIR=%PROJECT_DIR%\stock-market-predictor-env"
set "LOG_FILE=%SCRIPT_DIR%update_agent.log"

echo ======================================== >> "%LOG_FILE%"
echo [%date% %time%] Agent triggered. >> "%LOG_FILE%"

REM Activate virtualenv
call "%VENV_DIR%\Scripts\activate.bat"

REM Run the update from the project root so relative imports resolve
cd /d "%PROJECT_DIR%"
python -c "from backend.database.scripts.update_db import update_database; update_database()" >> "%LOG_FILE%" 2>&1
set EXIT_CODE=%ERRORLEVEL%

echo [%date% %time%] Finished with exit code %EXIT_CODE%. >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"

exit /b %EXIT_CODE%
