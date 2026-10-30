#!/bin/sh
# Start a VNC display and Blender on it, with server.py taking code from run.py. Returns
# once Blender listens; a second run while it does is a no-op.
HERE=$(cd "$(dirname "$0")" && pwd)
if python3 -c 'import socket; socket.create_connection(("127.0.0.1", 8765))' 2>/dev/null; then
    exit 0
fi

# The password the person's VNC viewer asks for. macOS Screen Sharing takes no server without one.
mkdir -p ~/.vnc
echo ailoy | vncpasswd -f > ~/.vnc/passwd
chmod 600 ~/.vnc/passwd
nohup Xvnc :0 -geometry 1600x900 -depth 24 -rfbport 5900 \
    -SecurityTypes VncAuth -PasswordFile ~/.vnc/passwd > /tmp/xvnc.log 2>&1 &
for i in $(seq 50); do [ -e /tmp/.X11-unix/X0 ] && break; sleep 0.2; done

DISPLAY=:0 nohup blender --gpu-backend vulkan -p 0 0 1600 900 -P "$HERE/server.py" \
    > /tmp/blender.log 2>&1 &
for i in $(seq 120); do
    grep -q "ailoy: listening" /tmp/blender.log && exit 0
    sleep 1
done
echo "Blender did not start; the end of /tmp/blender.log:" >&2
tail -40 /tmp/blender.log >&2
exit 1
