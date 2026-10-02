"""Take screenshots of the viewer's display for whoever asks.

    python3 shot.py PORT

The agent plays from a console of its own, which has no display to take a picture of. So
it connects here, at the port this console published, and sends a name; this takes the screenshot into
`/artifacts/shots/NAME.png`, which both consoles see, and answers with its path.
"""

import os
import re
import socket
import subprocess
import sys
import threading

SHOTS = "/artifacts/shots"


def serve(conn):
    with conn:
        name = conn.makefile("rb").readline().decode().strip()
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "shot"
        path = os.path.join(SHOTS, f"{name}.png")
        os.makedirs(SHOTS, exist_ok=True)
        done = subprocess.run(
            ["import", "-window", "root", "-display", ":99", path],
            capture_output=True,
            timeout=30,
        )
        if done.returncode == 0:
            conn.sendall(f"ok {path}\n".encode())
        else:
            conn.sendall(f"error {done.stderr.decode().strip()}\n".encode())


def main():
    server = socket.create_server(("127.0.0.1", int(sys.argv[1])))
    while True:
        conn, _ = server.accept()
        threading.Thread(target=serve, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
