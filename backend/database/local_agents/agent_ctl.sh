#!/bin/bash
# agent_ctl.sh — Control the MarketLens background database update agent.
# Usage: ./agent_ctl.sh [start|stop|status|run|logs]

PLIST_NAME="com.marketlens.update-agent"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME.plist"
LOG_FILE="$SCRIPT_DIR/update_agent.log"

case "$1" in
    start)
        # Always recreate the plist to ensure absolute paths are up-to-date
        mkdir -p "$HOME/Library/LaunchAgents"
        
        cat <<EOF > "$PLIST_DST"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$PLIST_NAME</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$SCRIPT_DIR/update_agent.sh</string>
    </array>

    <!-- Run at 4:30 PM ET, Monday through Friday -->
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>16</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>16</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>16</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>16</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>16</integer><key>Minute</key><integer>30</integer></dict>
    </array>

    <key>StandardOutPath</key>
    <string>$LOG_FILE</string>

    <key>StandardErrorPath</key>
    <string>$LOG_FILE</string>
</dict>
</plist>
EOF

        # Reload it to apply new paths
        launchctl unload "$PLIST_DST" 2>/dev/null
        launchctl load "$PLIST_DST"
        echo "✅ Agent started. Database will update at 4:30 PM ET every weekday."
        ;;

    stop)
        if ! launchctl list "$PLIST_NAME" &>/dev/null; then
            echo "⚠️  Agent is not running."
            # Clean up symlink if it exists
            rm -f "$PLIST_DST"
            exit 0
        fi

        launchctl unload "$PLIST_DST"
        rm -f "$PLIST_DST"
        echo "🛑 Agent stopped."
        ;;

    status)
        if launchctl list "$PLIST_NAME" &>/dev/null; then
            echo "✅ Agent is loaded and scheduled."
            launchctl list "$PLIST_NAME"
        else
            echo "⚠️  Agent is not running."
        fi
        ;;

    run)
        echo "🚀 Running database update now..."
        bash "$SCRIPT_DIR/update_agent.sh"
        echo "✅ Manual run complete. Check logs: $LOG_FILE"
        ;;

    logs)
        if [[ ! -f "$LOG_FILE" ]]; then
            echo "⚠️  No log file found yet. Run the agent first."
            exit 1
        fi
        tail -f "$LOG_FILE"
        ;;

    *)
        echo "MarketLens Database Update Agent"
        echo ""
        echo "Usage: ./agent_ctl.sh <command>"
        echo ""
        echo "Commands:"
        echo "  start    Install and enable the background agent"
        echo "  stop     Disable and remove the background agent"
        echo "  status   Check if the agent is currently loaded"
        echo "  run      Manually trigger an immediate database update"
        echo "  logs     Tail the live agent log output"
        ;;
esac
