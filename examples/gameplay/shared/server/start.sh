#!/bin/sh
# Start OpenTTD as a dedicated server run by the bridge, a client watching it on a virtual
# display, and a VNC server showing that display.
#
#     sh start.sh WIDTHxHEIGHT PASSWORD [SAVEGAME]
#
# The host publishes the VNC server (5900), the game's admin port (3977) and `shot.py` (5902),
# at the same ports on its loopback but the viewer's. A new game unless SAVEGAME, a file in the saves folder, is given. Everything is left
# running when this returns; the server's process id is in /tmp/openttd.pid.
set -eu
SIZE=$1
PASSWORD=$2
SAVE=${3:-}
HERE=$(dirname "$0")
LOGS=/artifacts/logs

# Saved games, and the scripts the game finds, under XDG's data folder on the mount so that
# the saves outlast the console. The settings are written fresh each time, out of the way.
export XDG_DATA_HOME=/artifacts/user XDG_CONFIG_HOME=/tmp/config
DATA=$XDG_DATA_HOME/openttd
SERVER=$XDG_CONFIG_HOME/openttd
VIEWER=/tmp/viewer
mkdir -p "$LOGS" "$DATA/save" "$DATA/ai" "$DATA/game" "$SERVER" "$VIEWER"
rm -rf "$DATA/ai/AiloyCompany" "$DATA/game/AiloyBridge"
cp -r "$HERE/ai/AiloyCompany" "$DATA/ai/"
cp -r "$HERE/game/AiloyBridge" "$DATA/game/"

# The version is OpenTTD 14's own: without it the file is taken for an older one, from
# before private settings had files of their own, and `secrets.cfg` goes unread.
#
# The game is paused but for when the agent acts or waits, so it neither pauses for a
# client joining nor for there being none. A script gets far more work done in a tick than
# by default, so that finding a road takes game hours, not weeks.
cat >"$SERVER/openttd.cfg" <<EOF
[version]
ini_version = 7

[game_creation]
map_x = ${OPENTTD_MAP_X:-8}
map_y = ${OPENTTD_MAP_Y:-8}
starting_year = ${OPENTTD_YEAR:-1950}
landscape = temperate

[difficulty]
max_no_competitors = 0

[ai]
ai_in_multiplayer = true

[script]
script_max_opcode_till_suspend = 250000

[network]
pause_on_join = false
min_active_clients = 0
server_admin_port = 3977

[game_scripts]
AiloyBridge =
EOF
printf '[network]\nserver_name = ailoy\nclient_name = server\nparticipate_survey = no\n' \
    >"$SERVER/private.cfg"
printf '[network]\nadmin_password = %s\n' "${OPENTTD_ADMIN_PASSWORD:-ailoy}" >"$SERVER/secrets.cfg"

# The viewer's settings, next to its own `openttd.cfg`, and never written back.
printf '[version]\nini_version = 7\n\n[misc]\nresolution = %s,%s\nvideo_hw_accel = false\n' \
    "${SIZE%x*}" "${SIZE#*x}" >"$VIEWER/openttd.cfg"
printf '[network]\nclient_name = viewer\nparticipate_survey = no\n' >"$VIEWER/private.cfg"

if [ -n "$SAVE" ]; then
    GAME="-g $DATA/save/$SAVE"
else
    GAME="-g -G ${OPENTTD_SEED:-$(date +%s)}"
fi
# shellcheck disable=SC2086
nohup /usr/games/openttd -D 127.0.0.1:3979 $GAME -x -d script=4 >"$LOGS/server.log" 2>&1 &
echo $! >/tmp/openttd.pid

# Open the company, pause, and wait for the bridge to answer.
python3 "$HERE/ready.py" >"$LOGS/ready.log" 2>&1 || {
    echo "the game did not come up:" >&2
    tail -n 20 "$LOGS/ready.log" "$LOGS/server.log" >&2
    exit 1
}

export DISPLAY=:99
nohup Xvfb :99 -screen 0 "${SIZE}x24" -nolisten tcp >"$LOGS/xvfb.log" 2>&1 &
for _ in $(seq 50); do [ -S /tmp/.X11-unix/X99 ] && break; sleep 0.1; done

# No sound and no music: nothing would carry them to the viewer. `#255` joins as a
# spectator; a player can still open a company of their own from the game's menu.
nohup /usr/games/openttd -c "$VIEWER/openttd.cfg" -x -r "$SIZE" -s null -m null \
    -n "127.0.0.1:3979#255" >"$LOGS/viewer.log" 2>&1 &

# `-threads` so that input is taken while a frame is being sent: the game redraws all the
# time, and with one thread a click waits behind the frames, until the next input comes.
nohup x11vnc -display :99 -localhost -rfbport 5900 -forever -shared -passwd "$PASSWORD" \
    -threads -quiet >"$LOGS/x11vnc.log" 2>&1 &
nohup python3 "$HERE/shot.py" 5902 >"$LOGS/shot.log" 2>&1 &
