"""OpenTTD's admin port, as much of it as the bridge needs.

A packet is its size (two bytes, little-endian, counting these three), its type (one byte)
and its fields: integers little-endian, booleans one byte, strings ending in a zero byte.
"""

import json
import os
import socket
import struct
import time

# What this client sends.
ADMIN_JOIN = 0
ADMIN_QUIT = 1
ADMIN_UPDATE_FREQUENCY = 2
ADMIN_POLL = 3
ADMIN_RCON = 5
ADMIN_GAMESCRIPT = 6

# What the server sends.
SERVER_FULL = 100
SERVER_BANNED = 101
SERVER_ERROR = 102
SERVER_PROTOCOL = 103
SERVER_WELCOME = 104
SERVER_SHUTDOWN = 106
SERVER_DATE = 107
SERVER_RCON = 120
SERVER_CONSOLE = 121
SERVER_GAMESCRIPT = 124
SERVER_RCON_END = 125

# Where the game is: the same address from either console. In the game's own it is the
# server itself, and from the agent's it is the port the game's console published, which
# every console on the host reaches at 127.0.0.1.
ADMIN_ADDRESSES = [("127.0.0.1", 3977)]
SHOT_ADDRESSES = [("127.0.0.1", 5902)]

UPDATE_DATE = 0
UPDATE_GAMESCRIPT = 9
FREQ_DAILY = 2
FREQ_AUTOMATIC = 64


class AdminError(Exception):
    pass


class Reader:
    def __init__(self, data):
        self.data, self.at = data, 0

    def u8(self):
        self.at += 1
        return self.data[self.at - 1]

    def u16(self):
        self.at += 2
        return struct.unpack_from("<H", self.data, self.at - 2)[0]

    def u32(self):
        self.at += 4
        return struct.unpack_from("<I", self.data, self.at - 4)[0]

    def string(self):
        end = self.data.index(b"\0", self.at)
        s = self.data[self.at : end].decode("utf-8", "replace")
        self.at = end + 1
        return s


def string(s):
    return s.encode() + b"\0"


def connect(addresses, timeout):
    """A connection to the first of `addresses` that takes one, trying until `timeout`."""
    deadline = time.monotonic() + timeout
    while True:
        for address in addresses:
            try:
                return socket.create_connection(address, timeout=5)
            except OSError:
                pass
        if time.monotonic() > deadline:
            where = " or ".join(f"{h}:{p}" for h, p in addresses)
            raise AdminError(f"the game does not answer at {where}")
        time.sleep(0.5)


class Admin:
    """One session on the admin port: joined on construction, quit on `close`."""

    def __init__(self, password=None, timeout=10):
        password = password or os.environ.get("OPENTTD_ADMIN_PASSWORD", "ailoy")
        self.sock = connect(ADMIN_ADDRESSES, timeout)
        self.buf = b""
        self.send(ADMIN_JOIN, string(password) + string("ailoy") + string("1"))
        while True:
            got = self.recv(timeout=30)
            if got is None:
                raise AdminError("the game's admin port took the connection and said nothing")
            kind, r = got
            if kind == SERVER_WELCOME:
                r.string()  # server name
                r.string()  # revision
                r.u8()  # dedicated
                r.string()  # map name
                r.u32()  # seed
                r.u8()  # landscape
                r.u32()  # start date
                self.map_size = (r.u16(), r.u16())
                break
            if kind in (SERVER_ERROR, SERVER_FULL, SERVER_BANNED):
                raise AdminError(f"the admin port refused to join (packet {kind})")
        self.send(ADMIN_UPDATE_FREQUENCY, struct.pack("<HH", UPDATE_GAMESCRIPT, FREQ_AUTOMATIC))
        self.send(ADMIN_UPDATE_FREQUENCY, struct.pack("<HH", UPDATE_DATE, FREQ_DAILY))
        self.date = None

    def close(self):
        try:
            self.send(ADMIN_QUIT, b"")
            self.sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def send(self, kind, payload):
        self.sock.sendall(struct.pack("<HB", len(payload) + 3, kind) + payload)

    def recv(self, timeout=None):
        """The next packet as (type, Reader), or None if `timeout` passes first."""
        self.sock.settimeout(timeout)
        while True:
            if len(self.buf) >= 3:
                size, kind = struct.unpack_from("<HB", self.buf)
                if len(self.buf) >= size:
                    payload, self.buf = self.buf[3:size], self.buf[size:]
                    r = Reader(payload)
                    if kind == SERVER_DATE:
                        self.date = r.u32()
                        r.at = 0
                    if kind == SERVER_SHUTDOWN:
                        raise AdminError("the game has shut down")
                    return kind, r
            try:
                data = self.sock.recv(65536)
            except socket.timeout:
                return None
            if not data:
                raise AdminError("the game closed the admin connection")
            self.buf += data

    def rcon(self, command):
        """Run a console command, and return what it printed."""
        self.send(ADMIN_RCON, string(command))
        lines = []
        while True:
            got = self.recv(timeout=30)
            if got is None:
                raise AdminError(f"no answer to the console command {command!r}")
            kind, r = got
            if kind == SERVER_RCON:
                r.u16()  # colour
                lines.append(r.string())
            elif kind == SERVER_RCON_END:
                return "\n".join(lines)

    def gamescript(self, request, timeout=600):
        """Send `request` to the bridge and gather its answer.

        The bridge answers in one or more messages with the request's `id`: each `part` a
        list of items, and the last `done`, with the rest of the result.
        """
        self.send(ADMIN_GAMESCRIPT, string(json.dumps(request, separators=(",", ":"))))
        items = []
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            got = self.recv(timeout=max(left, 0.1))
            if got is None:
                raise AdminError(f"the bridge did not answer {request.get('cmd')!r} in {timeout} s")
            kind, r = got
            if kind != SERVER_GAMESCRIPT:
                continue
            msg = json.loads(r.string())
            if msg.get("id") != request["id"]:
                continue
            if "part" in msg:
                items.extend(msg["part"])
            if msg.get("done", True):
                if items:
                    msg.setdefault("result", {})["items"] = items
                return msg

    def poll_date(self):
        """The game's date, in days."""
        self.date = None
        self.send(ADMIN_POLL, struct.pack("<BI", UPDATE_DATE, 0))
        deadline = time.monotonic() + 10
        while self.date is None:
            if time.monotonic() > deadline:
                raise AdminError("the game did not say its date")
            self.recv(timeout=1)
        return self.date

    def wait_date(self, until, timeout):
        """Read packets until the game's date is `until` or later, or `timeout` passes."""
        deadline = time.monotonic() + timeout
        while self.date is None or self.date < until:
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            self.recv(timeout=left)
        return True
