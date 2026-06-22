import errno
import json
import os
import sys
import urllib.parse
import urllib.request

from mfusepy import FUSE, FuseOSError, Operations

BASE = os.environ["VFS_HOST"].rstrip("/")
TOKEN = os.environ.get("VFS_TOKEN", "")


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

    def getattr(self, path, fh=None):
        if path in self.wbuf:
            return self._attr(False, len(self.wbuf[path]))
        st = json.loads(_request("GET", "/stat", path).decode())
        if not st.get("exists"):
            raise FuseOSError(errno.ENOENT)
        return self._attr(st["is_dir"], st.get("size", 0))

    def _attr(self, is_dir, size):
        mode = 0o040755 if is_dir else 0o100644
        return {"st_mode": mode, "st_nlink": 2 if is_dir else 1,
                "st_size": size, "st_uid": os.getuid(), "st_gid": os.getgid(),
                "st_mtime": 0, "st_atime": 0, "st_ctime": 0}

    def readdir(self, path, fh):
        d = json.loads(_request("GET", "/readdir", path).decode())
        return [".", ".."] + [e["name"] for e in d["entries"]]

    def open(self, path, flags):
        return 0

    def create(self, path, mode, fi=None):
        self.wbuf[path] = bytearray()
        return 0

    def read(self, path, size, offset, fh):
        data = self.rcache.get(path)
        if data is None:
            data = _request("GET", "/read", path,
                            query={"offset": offset, "size": size})
            return bytes(data)
        return bytes(data[offset:offset + size])

    def write(self, path, data, offset, fh):
        buf = self.wbuf.setdefault(path, bytearray())
        if offset > len(buf):
            buf.extend(b"\x00" * (offset - len(buf)))
        buf[offset:offset + len(data)] = data
        return len(data)

    def truncate(self, path, length, fh=None):
        buf = self.wbuf.get(path)
        if buf is None:
            try:
                buf = bytearray(_request("GET", "/read", path,
                                         query={"offset": 0, "size": length}))
            except Exception:
                buf = bytearray()
            self.wbuf[path] = buf
        del buf[length:]

    def flush(self, path, fh):
        self._put(path)
        return 0

    def release(self, path, fh):
        self._put(path)
        return 0

    def _put(self, path):
        if path not in self.wbuf:
            return
        _request("PUT", "/write", path, body=bytes(self.wbuf[path]))
        del self.wbuf[path]
        self.rcache.pop(path, None)


if __name__ == "__main__":
    FUSE(Forward(), sys.argv[1], foreground=True, nothreads=True)
