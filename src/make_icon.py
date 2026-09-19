#!/usr/bin/env python3
"""Build the app icon set from the master art. Standard library only.

    python3 src/make_icon.py [SOURCE]

SOURCE defaults to `src/assets/icon-source.png`, the master art. The script
crops it to the artwork, turns the surrounding white into real transparency,
and writes `icon-16/32/64/128/256/512.png` beside it as RGBA.

Why this is code rather than six files someone exported once: a retained
artifact whose producer is missing cannot be corrected later. Re-crop, re-cut or
re-scale by editing this and running it, not by hand-editing PNGs.

Pillow is deliberately not a dependency -- the engine is standard-library-only
and an icon is not a reason to change that. PNG is zlib plus five row filters,
which is a page of code, and `sips` (built into macOS) handles JPEG if the
master ever arrives in that form.

The white is removed by FLOOD FILL from the border, not by keying every white
pixel. The artwork contains near-white of its own -- the page the seal is
resting on -- and keying on colour alone punches holes straight through it.
"""
from __future__ import annotations
import collections, pathlib, struct, subprocess, sys, zlib

HERE = pathlib.Path(__file__).parent
ASSETS = HERE / "assets"
SIZES = (16, 32, 64, 128, 256, 512)
WHITE = 236          # >= this on every channel counts as background
FEATHER = 1.0        # px of edge softening at master scale, before downsampling


# ------------------------------------------------------------------- png ---
def read_png(path: pathlib.Path):
    """-> (width, height, RGB bytearray). Handles colour types 2 and 6."""
    d = path.read_bytes()
    if d[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"{path} is not a PNG")
    idat = bytearray()
    pos, w, h, ct = 8, None, None, None
    while pos < len(d):
        n = struct.unpack(">I", d[pos:pos + 4])[0]
        tag = d[pos + 4:pos + 8]
        body = d[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            w, h, bd, ct = struct.unpack(">IIBB", body[:10])
            if bd != 8 or ct not in (2, 6):
                raise SystemExit(f"{path}: need 8-bit RGB or RGBA, got {bd}/{ct}")
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
        pos += 12 + n
    stride_in = w * (3 if ct == 2 else 4)
    raw = zlib.decompress(bytes(idat))
    out = bytearray(w * h * 3)
    prev = bytearray(stride_in)
    bpp = 3 if ct == 2 else 4
    p = 0
    for y in range(h):
        f = raw[p]; p += 1
        line = bytearray(raw[p:p + stride_in]); p += stride_in
        # undo the row filter
        if f == 1:
            for i in range(bpp, stride_in):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif f == 2:
            for i in range(stride_in):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif f == 3:
            for i in range(stride_in):
                a = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif f == 4:
            for i in range(stride_in):
                a = line[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        elif f != 0:
            raise SystemExit(f"{path}: unknown row filter {f}")
        prev = line
        for x in range(w):
            s, t = x * bpp, (y * w + x) * 3
            out[t:t + 3] = line[s:s + 3]
    return w, h, out


def write_png(path: pathlib.Path, w: int, h: int, rgba: bytearray):
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw += rgba[y * w * 4:(y + 1) * w * 4]

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
                     + chunk(b"IEND", b""))


# ----------------------------------------------------------------- shape ---
def background_alpha(w: int, h: int, rgb: bytearray) -> bytearray:
    """Alpha 0 for background reachable from the border, 255 for artwork.

    Flood fill, not a colour key: the artwork has its own near-white page, and
    keying every white pixel punches holes through it.
    """
    alpha = bytearray(b"\xff" * (w * h))
    seen = bytearray(w * h)
    q = collections.deque()

    def white(i):
        p = i * 3
        return rgb[p] >= WHITE and rgb[p + 1] >= WHITE and rgb[p + 2] >= WHITE

    for x in range(w):
        for i in (x, (h - 1) * w + x):
            if not seen[i] and white(i):
                seen[i] = 1; alpha[i] = 0; q.append(i)
    for y in range(h):
        for i in (y * w, y * w + w - 1):
            if not seen[i] and white(i):
                seen[i] = 1; alpha[i] = 0; q.append(i)

    while q:
        i = q.popleft()
        x, y = i % w, i // w
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h:
                j = ny * w + nx
                if not seen[j] and white(j):
                    seen[j] = 1; alpha[j] = 0; q.append(j)
    return alpha


def crop(w: int, h: int, rgb: bytearray, alpha: bytearray):
    xs0, ys0, xs1, ys1 = w, h, -1, -1
    for y in range(h):
        row = y * w
        for x in range(w):
            if alpha[row + x]:
                if x < xs0: xs0 = x
                if x > xs1: xs1 = x
                if y < ys0: ys0 = y
                if y > ys1: ys1 = y
    nw, nh = xs1 - xs0 + 1, ys1 - ys0 + 1
    r2, a2 = bytearray(nw * nh * 3), bytearray(nw * nh)
    for y in range(nh):
        src = ((y + ys0) * w + xs0)
        r2[y * nw * 3:(y + 1) * nw * 3] = rgb[src * 3:(src + nw) * 3]
        a2[y * nw:(y + 1) * nw] = alpha[src:src + nw]
    return nw, nh, r2, a2


def tile_colour(w: int, h: int, rgb: bytearray) -> tuple[int, int, int]:
    """The tile's own background, taken from a band just inside its left edge."""
    from collections import Counter
    c = Counter()
    for y in range(h // 3, 2 * h // 3, 7):
        for x in range(int(w * 0.03), int(w * 0.08)):
            p = (y * w + x) * 3
            c[(rgb[p], rgb[p + 1], rgb[p + 2])] += 1
    return c.most_common(1)[0][0]


def clean_tile(w: int, h: int, rgb: bytearray, radius_frac: float = 0.214):
    """Re-impose a true rounded rectangle and repair the art's own edge.

    The generated master is not a clean tile: on one row its right edge stops
    ~17px short and white shows through, which downsamples into a visible bite
    out of the icon. Inheriting that would bake a rendering artifact into every
    size. Taking the shape from geometry instead makes the corners exact and the
    bite disappear, and any pixel inside the shape that is still background white
    is filled with the tile's own colour.

    Returns an anti-aliased alpha, softened over one pixel so the 512px output
    -- where the box filter only averages a 3x3 block -- is not stair-stepped.
    """
    fill = tile_colour(w, h, rgb)
    r = radius_frac * min(w, h)
    alpha = bytearray(w * h)
    for y in range(h):
        dy = max(r - y, y - (h - 1 - r), 0.0)
        row = y * w
        for x in range(w):
            dx = max(r - x, x - (w - 1 - r), 0.0)
            d = (dx * dx + dy * dy) ** 0.5
            cov = 1.0 if d <= r - 0.5 else (0.0 if d >= r + 0.5 else r + 0.5 - d)
            if cov <= 0.0:
                continue
            alpha[row + x] = int(cov * 255)
            p = (row + x) * 3
            if rgb[p] >= WHITE and rgb[p + 1] >= WHITE and rgb[p + 2] >= WHITE:
                rgb[p], rgb[p + 1], rgb[p + 2] = fill      # the bite
    return alpha, fill


def resample(w: int, h: int, rgb: bytearray, alpha: bytearray, size: int) -> bytearray:
    """Box filter, premultiplied so the transparent edge does not darken."""
    out = bytearray(size * size * 4)
    for oy in range(size):
        y0, y1 = oy * h // size, max(oy * h // size + 1, (oy + 1) * h // size)
        for ox in range(size):
            x0, x1 = ox * w // size, max(ox * w // size + 1, (ox + 1) * w // size)
            r = g = b = a = n = 0
            for y in range(y0, y1):
                base = y * w
                for x in range(x0, x1):
                    i = base + x
                    av = alpha[i]
                    p = i * 3
                    r += rgb[p] * av; g += rgb[p + 1] * av; b += rgb[p + 2] * av
                    a += av; n += 1
            t = (oy * size + ox) * 4
            if a:
                out[t] = min(255, r // a)
                out[t + 1] = min(255, g // a)
                out[t + 2] = min(255, b // a)
            out[t + 3] = a // n if n else 0
    return out


def main():
    if len(sys.argv) > 1:
        src = pathlib.Path(sys.argv[1])
    else:
        # The committed master is the JPEG as delivered; the PNG beside it is a
        # derived convenience and is not tracked.
        src = next((p for p in (ASSETS / "icon-source.png",
                                ASSETS / "icon-source.jpeg",
                                ASSETS / "icon-source.jpg") if p.is_file()), None)
    if src is None or not src.is_file():
        raise SystemExit(f"no master art in {ASSETS}")
    if src.suffix.lower() in (".jpg", ".jpeg"):
        png = src.with_suffix(".png")
        subprocess.run(["sips", "-s", "format", "png", str(src), "--out", str(png)],
                       check=True, capture_output=True)
        src = png

    w, h, rgb = read_png(src)
    # The flood fill is used ONLY to find the artwork's extent. The shape itself
    # then comes from geometry, so the master's own ragged edge cannot survive.
    alpha = background_alpha(w, h, rgb)
    w, h, rgb, alpha = crop(w, h, rgb, alpha)
    alpha, fill = clean_tile(w, h, rgb)
    print(f"  master {w}x{h} cropped; tile colour #{fill[0]:02x}{fill[1]:02x}{fill[2]:02x}")

    ASSETS.mkdir(parents=True, exist_ok=True)
    for size in SIZES:
        p = ASSETS / (f"icon-{size}.png")
        write_png(p, size, size, resample(w, h, rgb, alpha, size))
        print(f"  {p.relative_to(HERE.parent)}  {p.stat().st_size:,}B")
    # icon.png is the one other things point at.
    (ASSETS / "icon.png").write_bytes((ASSETS / "icon-256.png").read_bytes())
    print(f"  src/assets/icon.png  (copy of icon-256.png)")

    # macOS .icns, for when this becomes a real bundle. iconutil is built in, so
    # this costs nothing and means the packaging step has one less thing to
    # invent later. Non-macOS just skips it.
    if sys.platform == "darwin":
        iconset = ASSETS / "summer.iconset"
        iconset.mkdir(exist_ok=True)
        for size in SIZES:
            if size <= 512:
                (iconset / f"icon_{size}x{size}.png").write_bytes(
                    (ASSETS / f"icon-{size}.png").read_bytes())
            if size * 2 in SIZES:
                (iconset / f"icon_{size}x{size}@2x.png").write_bytes(
                    (ASSETS / f"icon-{size * 2}.png").read_bytes())
        r = subprocess.run(["iconutil", "-c", "icns", str(iconset),
                            "-o", str(ASSETS / "summer.icns")],
                           capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  src/assets/summer.icns  "
                  f"{(ASSETS / 'summer.icns').stat().st_size:,}B")
        else:
            print(f"  iconutil: {r.stderr.strip()[:80]}", file=sys.stderr)
        for p in iconset.iterdir():
            p.unlink()
        iconset.rmdir()
    return 0


if __name__ == "__main__":
    sys.exit(main())
