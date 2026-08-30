# MarketLens Background Update Agent

## What This Does
This is a background agent that **automatically updates the `screener_data.db` database** with the latest market data every weekday at **4:30 PM ET** (right after the U.S. stock market closes). It runs silently in the background — no terminal windows needed, no manual commands to remember.

Under the hood, it triggers the existing [`update_db.py`](file:///Users/liangchu/side%20projects/stock-market-predictor/database/update_db.py) script on a daily schedule using:
- **macOS:** `launchd` (Apple's native process scheduler)
- **Windows:** Task Scheduler (`schtasks`)

---

## Files Overview

### macOS Files

| File | Purpose |
|---|---|
| [`agent_ctl.sh`](file:///Users/liangchu/side%20projects/stock-market-predictor/backend/database/local_agents/agent_ctl.sh) | CLI control script — start, stop, and manage the agent |
| [`update_agent.sh`](file:///Users/liangchu/side%20projects/stock-market-predictor/backend/database/local_agents/update_agent.sh) | Wrapper script that activates the virtualenv and runs the database update |
| [`com.marketlens.update-agent.plist`](file:///Users/liangchu/side%20projects/stock-market-predictor/backend/database/local_agents/com.marketlens.update-agent.plist) | `launchd` service configuration (schedule, paths, logging) |

### Windows Files

| File | Purpose |
|---|---|
| [`agent_ctl.bat`](file:///Users/liangchu/side%20projects/stock-market-predictor/backend/database/local_agents/agent_ctl.bat) | CLI control script — start, stop, and manage the agent |
| [`update_agent.bat`](file:///Users/liangchu/side%20projects/stock-market-predictor/backend/database/local_agents/update_agent.bat) | Wrapper script that activates the virtualenv and runs the database update |

### Shared

| File | Purpose |
|---|---|
| `update_agent.log` | Auto-generated log file with timestamped output from each run |

---

## Getting Started (macOS)

### 1. Make the Scripts Executable
Run this once from the `backend/database/local_agents/` directory:
```bash
chmod +x agent_ctl.sh update_agent.sh
```

### 2. Start the Agent
```bash
./agent_ctl.sh start
```
You should see:
```
✅ Agent started. Database will update at 4:30 PM ET every weekday.
```

That's it. The agent is now running in the background and will persist across reboots.

---

## Getting Started (Windows)

### 1. Open a Terminal
Open **Command Prompt** or **PowerShell** and navigate to the `backend\database\local_agents\` directory:
```cmd
cd path\to\stock-market-predictor\backend\database\local_agents
```

### 2. Start the Agent
```cmd
agent_ctl.bat start
```
You should see:
```
[OK] Agent started. Database will update at 4:30 PM every weekday.
```

That's it. The agent is now registered with Windows Task Scheduler and will persist across reboots.

> **Note:** If you see a permissions error, try running the terminal as **Administrator** (right-click → "Run as administrator").

---

## Commands Reference

All commands are run from the `backend/database/local_agents/` directory.

### macOS

| Command | Action |
|---|---|
| `./agent_ctl.sh start` | Install and enable the background agent |
| `./agent_ctl.sh stop` | Disable and remove the background agent |
| `./agent_ctl.sh status` | Check if the agent is currently loaded |
| `./agent_ctl.sh run` | Manually trigger an immediate database update |
| `./agent_ctl.sh logs` | Tail the live agent log output |

### Windows

| Command | Action |
|---|---|
| `agent_ctl.bat start` | Create and enable the scheduled task |
| `agent_ctl.bat stop` | Remove the scheduled task |
| `agent_ctl.bat status` | Check if the scheduled task exists |
| `agent_ctl.bat run` | Manually trigger an immediate database update |
| `agent_ctl.bat logs` | Tail the live agent log output |

---

## How It Works

### macOS (`launchd`)

1. **macOS `launchd`** reads the `.plist` configuration and schedules the job for 4:30 PM ET every weekday.
2. At the scheduled time, `launchd` executes `update_agent.sh`.
3. The wrapper script activates the project's Python virtualenv (`stock-market-predictor-env/`).
4. It runs the `update_database()` function from [`update_db.py`](file:///Users/liangchu/side%20projects/stock-market-predictor/database/update_db.py), which:
   - Fetches the latest benchmark indices and their constituents
   - Downloads 1-year price history for all tickers
   - Fetches the latest news headlines
   - Writes everything to `screener_data.db`
5. All output is logged to `update_agent.log` with timestamps.

### Windows (Task Scheduler)

1. **Windows Task Scheduler** registers a weekly task named `MarketLens-UpdateAgent` that runs at 4:30 PM every weekday (Mon–Fri).
2. At the scheduled time, Task Scheduler executes `update_agent.bat`.
3. The wrapper script activates the project's Python virtualenv (`stock-market-predictor-env\Scripts\activate.bat`).
4. It runs the same `update_database()` function, performing the identical data refresh.
5. All output is logged to `update_agent.log` with timestamps.

---

## FAQ

### What happens on weekends and market holidays?
The agent only runs Monday through Friday. If a weekday is a market holiday (e.g., Christmas), the script will still run, but since the market was closed, it will fetch the same closing prices as the previous day. The `INSERT OR REPLACE` logic in SQLite handles this safely — no duplicates are created.

### What if my Mac is asleep at 4:30 PM?
`launchd` will automatically run the missed job as soon as your Mac wakes up. You won't miss an update.

### What if my Windows PC is asleep at 4:30 PM?
By default, Windows Task Scheduler will run the missed task the next time the PC is awake. To enable "wake to run," open Task Scheduler, find the `MarketLens-UpdateAgent` task, go to Properties → Conditions, and check "Wake the computer to run this task."

### Does this survive reboots?
- **macOS:** Yes. The plist is installed in `~/Library/LaunchAgents/`.
- **Windows:** Yes. The task is registered in Windows Task Scheduler which persists across reboots.

### Do I need to keep a terminal window open?
No. On both platforms, the agent runs entirely in the background as a system service.

### How do I know if something went wrong?
Run `./agent_ctl.sh logs` (macOS) or `agent_ctl.bat logs` (Windows) to check the output. Any errors from the Python script will be logged with full tracebacks.
