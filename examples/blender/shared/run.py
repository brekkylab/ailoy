"""Run Python in the Blender the person is watching, and print what it printed.

    python3 run.py SCRIPT.py
    python3 run.py -c 'CODE'

The code runs in Blender's own Python, on its main thread, in a namespace kept between runs.
Blender is in another machine, reached at 127.0.0.1:8765: SCRIPT.py is read here and its text
sent. Exits 1, with the traceback, when the code raises.
"""

import json
import socket
import sys

if sys.argv[1:2] == ["-c"]:
    code = sys.argv[2]
elif len(sys.argv) == 2:
    with open(sys.argv[1]) as f:
        code = f.read()
else:
    sys.exit(__doc__)

try:
    conn = socket.create_connection(("127.0.0.1", 8765))
except ConnectionRefusedError:
    sys.exit("Blender is not listening. It is started with the run, not from here: say so and stop.")
# One line, with no half-close after it: a published port carries that as a close.
conn.sendall(json.dumps({"code": code}).encode() + b"\n")
data = b""
while chunk := conn.recv(1 << 16):
    data += chunk
answer = json.loads(data)
print(answer["output"], end="")
sys.exit(0 if answer["ok"] else 1)
