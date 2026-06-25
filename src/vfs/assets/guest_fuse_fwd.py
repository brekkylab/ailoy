import errno
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request

from mfusepy import FUSE, FuseOSError, Operations

BASE = os.environ["VFS_HOST"].rstrip("/")
TOKEN = os.environ.get("VFS_TOKEN", "")
# How long a readdir-populated attr survives. The point is to absorb the
# getattr storm the kernel issues right after a readdir (e.g. `ls -la`), so
# even a small value avoids one network /stat per entry. Tunable via env.
ATTR_TTL = float(os.environ.get("VFS_ATTR_TTL", "5"))


def _request(method, route, path, query=None, body=None):
    q = {"path": path}
    if query:
        q.update(query)
    url = f"{BASE}{route}?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("x-vfs-token", TOKEN)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


class Forward(Operations):
    def __init__(self):
        self.wbuf = {}
        self.rcache = {}
        # path -> (expiry_monotonic, is_dir, size); filled by readdir + getattr.
        self.acache = {}
        # Guards the dicts above. Never held across a network _request so FUSE
        # callbacks (run on multiple threads) issue provider calls concurrently.
        self.lock = threading.Lock()

    def getattr(self, path, fh=None):
        with self.lock:
            if path in self.wbuf:
                return self._attr(False, len(self.wbuf[path]))
            ent = self.acache.get(path)
            if ent is not None and ent[0] > time.monotonic():
                return self._attr(ent[1], ent[2])
        st = json.loads(_request("GET", "/stat", path).decode())
        if not st.get("exists"):
            raise FuseOSError(errno.ENOENT)
        is_dir, size = st["is_dir"], st.get("size", 0)
        with self.lock:
            self.acache[path] = (time.monotonic() + ATTR_TTL, is_dir, size)
        return self._attr(is_dir, size)

    def _attr(self, is_dir, size):
        mode = 0o040755 if is_dir else 0o100644
        return {"st_mode": mode, "st_nlink": 2 if is_dir else 1,
                "st_size": size, "st_uid": os.getuid(), "st_gid": os.getgid(),
                "st_mtime": 0, "st_atime": 0, "st_ctime": 0}

    def readdir(self, path, fh):
        d = json.loads(_request("GET", "/readdir", path).decode())
        entries = d["entries"]
        # Cache each child's attrs so the kernel's follow-up getattr storm is
        # served locally instead of one /stat round trip per entry. readdir
        # already carries name/is_dir/size, so this costs nothing extra.
        base = path.rstrip("/")
        exp = time.monotonic() + ATTR_TTL
        with self.lock:
            for e in entries:
                child = f"{base}/{e['name']}" if base else f"/{e['name']}"
                self.acache[child] = (exp, e.get("is_dir", False),
                                      e.get("size", 0))
        return [".", ".."] + [e["name"] for e in entries]

    def open(self, path, flags):
        return 0

    def create(self, path, mode, fi=None):
        with self.lock:
            self.wbuf[path] = bytearray()
        return 0

    def read(self, path, size, offset, fh):
        with self.lock:
            data = self.rcache.get(path)
        if data is None:
            # Fetch the whole object once and serve every chunk from it. With
            # direct_io the kernel reads until a short read signals EOF, so a
            # stable buffer gives deterministic EOF and avoids re-fetching (and,
            # for rendered files like Notion page.json, re-rendering) per chunk.
            data = _request("GET", "/read", path)
            with self.lock:
                self.rcache[path] = data
        return bytes(data[offset:offset + size])

    def write(self, path, data, offset, fh):
        with self.lock:
            buf = self.wbuf.setdefault(path, bytearray())
            if offset > len(buf):
                buf.extend(b"\x00" * (offset - len(buf)))
            buf[offset:offset + len(data)] = data
            return len(data)

    def truncate(self, path, length, fh=None):
        with self.lock:
            buf = self.wbuf.get(path)
        if buf is None:
            try:
                data = _request("GET", "/read", path,
                                query={"offset": 0, "size": length})
            except Exception:
                data = b""
            buf = bytearray(data)
            with self.lock:
                self.wbuf[path] = buf
        with self.lock:
            del buf[length:]

    def flush(self, path, fh):
        self._put(path)
        return 0

    def release(self, path, fh):
        self._put(path)
        # Drop the read buffer so a re-open re-fetches (and memory is bounded).
        with self.lock:
            self.rcache.pop(path, None)
        return 0

    def _put(self, path):
        with self.lock:
            if path not in self.wbuf:
                return
            payload = bytes(self.wbuf[path])
        _request("PUT", "/write", path, body=payload)
        with self.lock:
            self.wbuf.pop(path, None)
            self.rcache.pop(path, None)
            self.acache.pop(path, None)


if __name__ == "__main__":
    # direct_io: reads go straight to the read handler, so the kernel does not
    # clamp them to the stat size. Listings may report size 0/unknown for
    # dynamically rendered files (e.g. Notion page.json); the handler returns
    # the real bytes and signals EOF via a short read.
    FUSE(Forward(), sys.argv[1], foreground=True, nothreads=False,
         direct_io=True)
