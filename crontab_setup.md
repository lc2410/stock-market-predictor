# Oracle Cloud Daily Automation Setup

Since you are using Oracle Cloud (Always Free tier), which runs a Linux VM (likely Oracle Linux or Ubuntu), the standard and completely free way to automate the daily database updates is by using `cron`.

Here is the exact setup you need to run on your Oracle Cloud VM to run the background script automatically at 4:30 PM EST every day (after the market closes).

### Step 1: Find your timezone
First, check what timezone your Oracle Cloud VM is in by running:
```bash
date
```
If your server is in UTC (which is common for cloud VMs), 4:30 PM EST is 9:30 PM UTC (or 8:30 PM UTC during Daylight Saving Time). It is highly recommended to set your server timezone to EST/EDT to avoid DST confusion:
```bash
sudo timedatectl set-timezone America/New_York
```

### Step 2: Open the Crontab Editor
Run this command to edit your user's crontab file:
```bash
crontab -e
```

### Step 3: Add the Cron Job
Add the following line to the bottom of the file. This tells cron to run the update script at 16:30 (4:30 PM) every day from Monday to Friday.

```bash
30 16 * * 1-5 cd /path/to/your/project/backend && /path/to/your/virtualenv/bin/python database/scripts/update_db.py >> /path/to/your/project/backend/update_db.log 2>&1
```

**Important Replacements:**
1. Replace `/path/to/your/project/backend` with the absolute path to your project's backend folder.
2. Replace `/path/to/your/virtualenv/bin/python` with the path to the Python executable where you installed the dependencies. (You can find this by running `which python` while your virtual environment is activated).

### Example
If your project is located at `/home/opc/stock-market-predictor` and your venv is in `/home/opc/stock-market-predictor/backend/venv`, the line would look like this:

```bash
30 16 * * 1-5 cd /home/opc/stock-market-predictor/backend && /home/opc/stock-market-predictor/backend/venv/bin/python database/scripts/update_db.py >> /home/opc/stock-market-predictor/backend/cron.log 2>&1
```

### Step 4: Verify
To verify the cron job was installed correctly, you can list your active cron jobs:
```bash
crontab -l
```

### What about weekends and holidays?
The `1-5` part of the cron schedule ensures it only runs Monday through Friday. If a weekday happens to be a market holiday (e.g., Christmas Day), the script will still run, but since the market was closed, it will simply fetch the exact same closing prices as the day before. The SQLite `INSERT OR REPLACE` logic ensures it safely handles this without creating duplicate entries. It's completely safe and keeps the logic simple!
