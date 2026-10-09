"""Tests for the deployed Space app template (`app.py.tmpl`).

The Space app is Python, outside the Rust cargo suite; run it directly with
any interpreter that has fastapi, httpx, and huggingface_hub installed
(e.g. `python3 src/space_template/test_app.py`). Tests skip themselves when
the dependencies are missing.

Focus: HTTP Range handling on the public asset endpoint. The backing
`bricks.bin` can be multi-GB, and the route is reachable by anyone who can
load the Space — so a Range request must never buffer more than `_CHUNK` of
it in RAM per request (memory-exhaustion DoS otherwise).
"""

import builtins
import importlib.util
import os
import tempfile
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    from fastapi.testclient import TestClient  # noqa: F401
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False


def load_app():
    """Render app.py.tmpl with a fake bucket id and import it as a module."""
    with open(os.path.join(HERE, "app.py.tmpl")) as f:
        src = f.read().replace("__BUCKET_ID__", "owner/test-bucket")
    d = tempfile.mkdtemp(prefix="arbvis-space-app-")
    path = os.path.join(d, "space_app.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("space_app", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, d


class TrackingReader:
    """File wrapper that records the sizes of `read` calls it sees."""

    def __init__(self, f, reads):
        self._f = f
        self._reads = reads

    def __enter__(self):
        self._f.__enter__()
        return self

    def __exit__(self, *exc):
        return self._f.__exit__(*exc)

    def seek(self, *a):
        return self._f.seek(*a)

    def read(self, n=None):
        data = self._f.read(n)
        self._reads.append(len(data))
        return data


class CountingSemaphore(threading.Semaphore):
    """Semaphore that records acquire calls so tests can assert pacing."""

    def __init__(self, value):
        super().__init__(value)
        self.acquires = 0

    def acquire(self, *a, **k):
        self.acquires += 1
        return super().acquire(*a, **k)


@unittest.skipUnless(_DEPS_OK, "needs fastapi + httpx")
class RangeStreamingTest(unittest.TestCase):
    class _FakeFS:
        """Hub stand-in: serves the test bytes without touching the network."""

        def __init__(self, data):
            self.data = data

        def open(self, path, mode="rb", block_size=None):
            import io

            class Ctx(io.BytesIO):
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    self.close()

            return Ctx(self.data)

        def size(self, path):
            return len(self.data)

    def setUp(self):
        self.mod, _tmpdir = load_app()
        self.client = TestClient(self.mod.app)
        self.mirror = tempfile.mkdtemp(prefix="arbvis-space-mirror-")
        self.mod._MIRROR_DIR = self.mirror
        self.data = os.urandom(self.mod._CHUNK + self.mod._CHUNK // 2)  # 1.5 chunks
        with open(os.path.join(self.mirror, "bricks.bin"), "wb") as f:
            f.write(self.data)
        size = len(self.data)
        # Pre-seed the size cache so no Hub metadata call happens in the test.
        self.mod._size_cache["hf://buckets/owner/test-bucket/bricks.bin"] = size
        self.mod._fs = self._FakeFS(self.data)

    def test_open_range_never_buffers_more_than_chunk(self):
        reads = []
        real_open = builtins.open

        def tracking_open(file, mode="r", *a, **k):
            f = real_open(file, mode, *a, **k)
            if os.path.basename(str(file)) == "bricks.bin" and "b" in mode:
                return TrackingReader(f, reads)
            return f

        builtins.open = tracking_open
        try:
            r = self.client.get("/bricks.bin", headers={"Range": "bytes=0-"})
        finally:
            builtins.open = real_open

        self.assertEqual(r.status_code, 206)
        self.assertEqual(r.content, self.data)
        self.assertEqual(r.headers["content-range"], "bytes 0-%d/%d" % (len(self.data) - 1, len(self.data)))
        # The hostile condition: a whole-file Range buffered in one read.
        self.assertLessEqual(max(reads), self.mod._CHUNK)
        self.assertGreater(len(reads), 1)
        # The explicit acquire must be released once the stream is drained.
        self.assertEqual(self.mod._HUB_SEM._value, 3)

    def test_hub_open_transient_error_releases_semaphore(self):
        # The Hub path acquires _HUB_SEM before fs().open; a non-404 failure
        # from that open (transient network error) must still release the slot.
        self.mod._start_mirror = lambda rest: None
        os.remove(os.path.join(self.mirror, "bricks.bin"))  # force the Hub path

        class BoomFS:
            def open(self, path, mode="rb", block_size=None):
                raise RuntimeError("transient network error")

        self.mod._fs = BoomFS()
        client = TestClient(self.mod.app, raise_server_exceptions=False)
        r = client.get("/bricks.bin", headers={"Range": "bytes=0-"})
        self.assertEqual(r.status_code, 500)
        self.assertEqual(self.mod._HUB_SEM._value, 3)

    def test_suffix_range_still_correct(self):
        r = self.client.get("/bricks.bin", headers={"Range": "bytes=-5"})
        self.assertEqual(r.status_code, 206)
        self.assertEqual(r.content, self.data[-5:])

    def test_no_range_streams_whole_file(self):
        r = self.client.get("/bricks.bin")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content, self.data)

    def test_no_range_serves_local_mirror_when_ready(self):
        # The mirror file is present (setUp wrote it), so a no-Range GET of
        # bricks.bin must come off local disk and never touch the Hub.
        opens = []

        class CountingFS(self._FakeFS):
            def open(self, *a, **k):
                opens.append(a[0])
                return super().open(*a, **k)

        self.mod._fs = CountingFS(self.data)
        r = self.client.get("/bricks.bin")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content, self.data)
        self.assertEqual(opens, [], "no-Range GET must serve the local mirror, not the Hub")

    def test_no_range_paces_hub_read_and_releases_semaphore(self):
        # Mirror absent: the no-Range GET must pace the Hub read through
        # _HUB_SEM and release the slot once the stream drains.
        self.mod._start_mirror = lambda rest: None
        os.remove(os.path.join(self.mirror, "bricks.bin"))
        self.mod._HUB_SEM = CountingSemaphore(3)
        r = self.client.get("/bricks.bin")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content, self.data)
        self.assertEqual(self.mod._HUB_SEM.acquires, 1)
        self.assertEqual(self.mod._HUB_SEM._value, 3)

    def test_no_range_kicks_mirror_download_when_not_ready(self):
        # Mirror absent: the no-Range GET must start the one-time background
        # mirror download, so repeated plain GETs do not keep the Space
        # hub-bound forever.
        os.remove(os.path.join(self.mirror, "bricks.bin"))
        kicks = []
        self.mod._start_mirror = lambda rest: kicks.append(rest)
        r = self.client.get("/bricks.bin")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(kicks, ["bricks.bin"])

    def test_no_range_missing_file_gets_404_not_500(self):
        # cached_size succeeded (the size cache is pre-seeded in setUp), so a
        # FileNotFoundError can only come from the open itself (asset vanished
        # between the metadata call and the open). That open happens inside
        # body() in the template... no: up front — so the client must see 404,
        # not a 200 with a truncated, error-terminated stream.
        class VanishedFS:
            def size(self, path):
                return len(vfs.data)

            def open(self, path, mode="rb", block_size=None):
                raise FileNotFoundError(path)

        vfs = self
        self.mod._fs = VanishedFS()
        self.mod._start_mirror = lambda rest: None
        os.remove(os.path.join(self.mirror, "bricks.bin"))  # force the Hub path
        client = TestClient(self.mod.app, raise_server_exceptions=False)
        r = client.get("/bricks.bin")
        self.assertEqual(r.status_code, 404)

    def test_oversized_range_digit_string_gets_400_not_500(self):
        # Python >=3.11 raises ValueError when int() sees more digits than the
        # interpreter's max-digit limit, which the template turns into 400. On
        # older interpreters the int parses fine: the open-ended form then has
        # start >= size (416), and the suffix form "bytes=-N" with N >= size
        # clamps to the whole file (206, correct per RFC 9110). The invariant
        # is only that the attacker-controlled header never surfaces as an
        # unhandled 500.
        r = self.client.get("/bricks.bin", headers={"Range": "bytes=" + "9" * 5000 + "-"})
        self.assertIn(r.status_code, (400, 416))
        r = self.client.get("/bricks.bin", headers={"Range": "bytes=-" + "9" * 5000})
        self.assertIn(r.status_code, (400, 416, 206))
        # A normal open-ended range still works after the hardening.
        r = self.client.get("/bricks.bin", headers={"Range": "bytes=0-"})
        self.assertEqual(r.status_code, 206)

    def test_path_traversal_segments_rejected(self):
        # A public visitor controls `rest`, which is interpolated into the
        # bucket path `hf://buckets/{BUCKET_ID}/{rest}` (and, for mirror
        # assets, joined onto local disk paths). Dot segments must never
        # reach those sinks: percent-encoded or literal, they get a 404.
        for p in ("/%2e%2e/other-bucket/file", "/..%2F..%2Fetc/passwd",
                  "/tiles/%2e/tile.avif", "/..%5C..%5Cfile", "/%2e/x",
                  # `rest` is a Starlette `:path` param, so an encoded leading
                  # slash decodes to a leading `/`; `os.path.join(_MIRROR_DIR,
                  # rest)` then discards the mirror directory. Must 404, and
                  # must not leave anything on disk outside the mirror.
                  "/%2Ftmp%2Fevil-pwn", "/%2Ftmp%2F%2e%2e%2Fevil"):
            r = self.client.get(p)
            self.assertEqual(r.status_code, 404, f"{p} must be rejected")
        # No Hub fetch was attempted for any of them (the fake FS serves
        # anything, so reaching it would return 200/206 instead), and the
        # absolute-rest attempts left nothing on disk outside the mirror.
        self.assertFalse(os.path.exists("/tmp/evil-pwn"))
        self.assertFalse(os.path.exists("/tmp/evil-pwn.part"))


    def test_traversal_free_asset_paths_still_served(self):
        r = self.client.get("/tiles/0/0/0.avif")
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
