#!/bin/sh
# Start OpenTTD on a virtual display, with a VNC server showing it.
#
#     sh start.sh WIDTHxHEIGHT HOST:PORT PASSWORD
#
# Everything is left running when this returns, and the game's process id is in
# /tmp/openttd.pid.
set -eu
SIZE=$1
HOST=$2
PASSWORD=$3
HERE=$(dirname "$0")
LOGS=/artifacts/logs

# OpenTTD keeps its settings and saved games under XDG's folders, which are on the mount
# so that they outlast the console.
export XDG_CONFIG_HOME=/artifacts/user XDG_DATA_HOME=/artifacts/user
CONFIG=$XDG_CONFIG_HOME/openttd
mkdir -p "$LOGS" "$CONFIG"

# The first time: a window as large as the display, no asking for a GPU there is not, and
# no asking whether to send a survey. OpenTTD fills in the rest when it quits. The version
# is OpenTTD 14's own: without it the file is taken for an older one, from before private
# settings had a file of their own, and `private.cfg` goes unread.
if [ ! -f "$CONFIG/openttd.cfg" ]; then
    printf '[misc]\nresolution = %s,%s\nvideo_hw_accel = false\n\n[version]\nini_version = 7\n' \
        "${SIZE%x*}" "${SIZE#*x}" >"$CONFIG/openttd.cfg"
fi
if [ ! -f "$CONFIG/private.cfg" ]; then
    printf '[network]\nparticipate_survey = no\n' >"$CONFIG/private.cfg"
fi

export DISPLAY=:99
nohup Xvfb :99 -screen 0 "${SIZE}x24" -nolisten tcp >"$LOGS/xvfb.log" 2>&1 &
for _ in $(seq 50); do [ -S /tmp/.X11-unix/X99 ] && break; sleep 0.1; done

# No sound and no music: nothing would carry them to the viewer.
nohup /usr/games/openttd -r "$SIZE" -s null -m null >"$LOGS/openttd.log" 2>&1 &
echo $! >/tmp/openttd.pid

# `-threads` so that input is taken while a frame is being sent: the game redraws all the
# time, and with one thread a click waits behind the frames, until the next input comes.
nohup x11vnc -display :99 -localhost -rfbport 5900 -forever -shared -passwd "$PASSWORD" \
    -threads -quiet >"$LOGS/x11vnc.log" 2>&1 &
nohup python3 "$HERE/tunnel.py" "$HOST" 5900 >"$LOGS/tunnel.log" 2>&1 &

sleep 3
if ! kill -0 "$(cat /tmp/openttd.pid)" 2>/dev/null; then
    echo "OpenTTD stopped:" >&2
    tail -n 20 "$LOGS/openttd.log" >&2
    exit 1
fi
