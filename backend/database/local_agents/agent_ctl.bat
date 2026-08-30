@echo off
REM agent_ctl.bat — Control the MarketLens background database update agent (Windows).
REM Usage: agent_ctl.bat [start|stop|status|run|logs]

set "TASK_NAME=MarketLens-UpdateAgent"
set "SCRIPT_DIR=%~dp0"
set "WRAPPER=%SCRIPT_DIR%update_agent.bat"
set "LOG_FILE=%SCRIPT_DIR%update_agent.log"

if "%~1"=="" goto usage

if /i "%~1"=="start" goto start
if /i "%~1"=="stop" goto stop
if /i "%~1"=="status" goto status
if /i "%~1"=="run" goto run
if /i "%~1"=="logs" goto logs
goto usage

:start
    REM Check if the task already exists
    schtasks /query /tn "%TASK_NAME%" >nul 2>&1
    if %ERRORLEVEL%==0 (
        echo [OK] Agent is already running.
        exit /b 0
    )

    if not exist "%WRAPPER%" (
        echo [ERROR] Wrapper script not found at %WRAPPER%
        exit /b 1
    )

    schtasks /create /tn "%TASK_NAME%" /tr "\"%WRAPPER%\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 16:30 /f >nul 2>&1
    if %ERRORLEVEL%==0 (
        echo [OK] Agent started. Database will update at 4:30 PM every weekday.
    ) else (
        echo [ERROR] Failed to create scheduled task. Try running as Administrator.
    )
    exit /b 0

:stop
    schtasks /query /tn "%TASK_NAME%" >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo [WARNING] Agent is not running.
        exit /b 0
    )

    schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1
    if %ERRORLEVEL%==0 (
        echo [STOPPED] Agent stopped.
    ) else (
        echo [ERROR] Failed to remove scheduled task. Try running as Administrator.
    )
    exit /b 0

:status
    schtasks /query /tn "%TASK_NAME%" /v /fo list 2>nul
    if %ERRORLEVEL% neq 0 (
        echo [WARNING] Agent is not running.
    )
    exit /b 0

:run
    echo [RUNNING] Running database update now...
    call "%WRAPPER%"
    echo [OK] Manual run complete. Check logs: %LOG_FILE%
    exit /b 0

:logs
    if not exist "%LOG_FILE%" (
        echo [WARNING] No log file found yet. Run the agent first.
        exit /b 1
    )
    REM Show the last 50 lines (Windows equivalent of tail)
    powershell -Command "Get-Content '%LOG_FILE%' -Tail 50 -Wait"
    exit /b 0

:usage
    echo MarketLens Database Update Agent (Windows)
    echo.
    echo Usage: agent_ctl.bat ^<command^>
    echo.
    echo Commands:
    echo   start    Create and enable the scheduled task
    echo   stop     Remove the scheduled task
    echo   status   Check if the scheduled task exists
    echo   run      Manually trigger an immediate database update
    echo   logs     Tail the live agent log output
    exit /b 0
