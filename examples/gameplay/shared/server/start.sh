#!/bin/sh
# Start OpenTTD as a dedicated server run by the bridge, and a client watching it on virtx's
# `display`, which serves its screen over VNC.
#
#     sh start.sh WIDTHxHEIGHT PASSWORD [SAVEGAME]
#
# VNC listens at 5900 and the game's admin port at 3977. A new game unless SAVEGAME, a file in
# the saves folder, is given. It prints `ready` once all of it is up, and runs until the game's
# server ends, so it is the one job the game is.
set -eu
SIZE=$1
PASSWORD=$2
SAVE=${3:-}
HERE=$(dirname "$0")
LOGS=/artifacts/logs

# Saved games, and the scripts the game finds, under XDG's data folder on the volume so that
# the saves outlast the sandbox. The settings are written fresh each time, out of the way.
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
# by default, so that finding a road takes game hours, not weeks. A save is written before
# `save` answers, so a file copied once it has is whole.
cat >"$SERVER/openttd.cfg" <<EOF
[version]
ini_version = 7

[gui]
threaded_saves = false

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
printf '[version]\nini_version = 7\n\n[misc]\nvideo_hw_accel = false\n' >"$VIEWER/openttd.cfg"
printf '[network]\nclient_name = viewer\nparticipate_survey = no\n' >"$VIEWER/private.cfg"

if [ -n "$SAVE" ]; then
    GAME="-g $DATA/save/$SAVE"
else
    GAME="-g -G ${OPENTTD_SEED:-$(date +%s)}"
fi
# shellcheck disable=SC2086
/usr/games/openttd -D 127.0.0.1:3979 $GAME -x -d script=4 >"$LOGS/server.log" 2>&1 &
SERVER_PID=$!

# Open the company, pause, and wait for the bridge to answer.
python3 "$HERE/ready.py" >"$LOGS/ready.log" 2>&1 || {
    echo "the game did not come up:" >&2
    tail -n 20 "$LOGS/ready.log" "$LOGS/server.log" >&2
    exit 1
}

# The viewer, on `display`'s screen `:99`, which `ttd.py look` takes its screenshots of. No
# sound and no music: nothing would carry them to the viewer. `#255` joins as a spectator; a
# player can still open a company of their own from the game's menu.
DISPLAY_VNC_PASSWORD=$PASSWORD display run --size "$SIZE" -- \
    /usr/games/openttd -c "$VIEWER/openttd.cfg" -x -r "$SIZE" -s null -m null \
    -n "127.0.0.1:3979#255" >"$LOGS/viewer.log" 2>&1 &
for _ in $(seq 100); do grep -qx ready "$LOGS/viewer.log" && break; sleep 0.1; done
grep -qx ready "$LOGS/viewer.log" || {
    echo "the viewer's screen did not come up:" >&2
    tail -n 20 "$LOGS/viewer.log" >&2
    exit 1
}

echo ready
wait "$SERVER_PID"
