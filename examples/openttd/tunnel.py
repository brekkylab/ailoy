"""Make a port in the console reachable from the host, which cannot connect in.

    python3 tunnel.py HOST:PORT LOCAL_PORT

A console can open a connection to a port on the host it was granted, and nothing on the host
can open one into the console. So this keeps one connection out to HOST:PORT, saying `CTRL`,
and each time the host writes `OPEN` on it, opens another saying `DATA` and joins that one to
LOCAL_PORT here. The host joins each `DATA` to a viewer that is waiting, and so a viewer on
the host talks to the server in here as if it had connected to it.
"""

import socket
import sys
import threading
import time


def pipe(src, dst):
    try:
        while data := src.recv(65536):
            dst.sendall(data)
    except OSError:
        pass
    for s in (src, dst):
        try:
            s.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def open_one(host, local):
    try:
        up = socket.create_connection(host)
        up.sendall(b"DATA\n")
        down = socket.create_connection(("127.0.0.1", local))
    except OSError as e:
        print(f"could not open a connection: {e}", flush=True)
        return
    threading.Thread(target=pipe, args=(up, down), daemon=True).start()
    pipe(down, up)


def main():
    host, port = sys.argv[1].rsplit(":", 1)
    host = (host, int(port))
    local = int(sys.argv[2])
    while True:
        try:
            with socket.create_connection(host) as ctrl:
                ctrl.sendall(b"CTRL\n")
                print(f"connected to {host[0]}:{host[1]}", flush=True)
                for line in ctrl.makefile("rb"):
                    if line.strip() == b"OPEN":
                        threading.Thread(target=open_one, args=(host, local), daemon=True).start()
        except OSError:
            pass
        time.sleep(2)


if __name__ == "__main__":
    main()
