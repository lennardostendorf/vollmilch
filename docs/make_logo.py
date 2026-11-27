"""Vollmilch pixel-art logo: a glass of milk on a 32x32 grid, plus a pixel-font wordmark.
python make_logo.py <outdir>"""
import sys

import numpy as np
from PIL import Image

PAL = {
    ".": None,
    "o": "#4a5a70",  # outline
    "g": "#d6e8f5",  # empty glass
    "G": "#bcd6ea",  # empty glass, shaded side
    "h": "#ffffff",  # highlight
    "t": "#ffffff",  # milk surface
    "l": "#e9e2cc",  # surface rim
    "m": "#fbf9f2",  # milk
    "s": "#ece7da",  # milk shade
    "d": "#d9d1bd",  # milk deep shade
    "b": "#a9c8df",  # glass base
    "B": "#8fb3cf",  # glass base, shaded side
    "w": "#fbf9f2",  # drop
    "W": "#d3cbb3",  # drop shade
    "x": "#c8d3df",  # table shadow
    "_": "#ffffff",  # banner background (so it also reads on dark pages)
}
N = 32
TOP, BOT, MILK = 6, 28, 12   # glass rows (outline included), milk surface row

g = np.full((N, N), ".", dtype="<U1")


def edges(y):  # tapered glass, wider at the rim
    f = (y - TOP) / (BOT - TOP)
    half = 9.4 - 3.2 * f
    return int(round(15.5 - half)), int(round(15.5 + half))


for y in range(TOP, BOT + 1):
    xl, xr = edges(y)
    g[y, xl] = g[y, xr] = "o"
    for x in range(xl + 1, xr):
        rel = (x - xl) / (xr - xl)
        if y == TOP or y == BOT:
            g[y, x] = "o"
        elif y >= BOT - 2:
            g[y, x] = "B" if rel > 0.75 else "b"
        elif y < MILK:
            g[y, x] = "G" if rel > 0.8 else "g"
        elif y == MILK:
            g[y, x] = "l" if (rel < 0.12 or rel > 0.88) else "t"
        elif y == MILK + 1:
            g[y, x] = "l" if rel > 0.85 else "t" if rel < 0.5 else "m"
        else:
            g[y, x] = "d" if rel > 0.88 else "s" if rel > 0.72 else "m"

# highlight streak on the left of the glass
for y in range(TOP + 2, BOT - 3):
    if y not in (MILK, MILK + 1, MILK + 6, MILK + 7):
        g[y, edges(y)[0] + 2] = "h"
for y in (TOP + 2, TOP + 3):
    g[y, edges(y)[0] + 3] = "h"

# falling drop above the glass (pointed top, round bottom)
DROP = ["..o..",
        ".owo.",
        "owwWo",
        "owWWo",
        ".oWo."]
for dy, row in enumerate(DROP):
    for dx, c in enumerate(row):
        if c != ".":
            g[dy, 20 + dx] = c

# shadow on the table
xl, xr = edges(BOT)
g[BOT + 1, xl + 1:xr] = "x"
g[BOT + 2, xl + 3:xr - 2] = "x"


def rgba(c):
    if PAL[c] is None:
        return (0, 0, 0, 0)
    h = PAL[c].lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,)


def to_img(grid, scale):
    a = np.array([[rgba(c) for c in row] for row in grid], np.uint8)
    return Image.fromarray(a, "RGBA").resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST)


def to_svg(grid, title):
    H, W = grid.shape
    rects = [f'<rect width="{W}" height="{H}" fill="{PAL["_"]}"/>'] if (grid == "_").any() else []
    for y in range(H):
        x = 0
        while x < W:  # merge horizontal runs of one colour
            c = grid[y, x]
            x2 = x
            while x2 + 1 < W and grid[y, x2 + 1] == c:
                x2 += 1
            if PAL[c] and c != "_":
                rects.append(f'<rect x="{x}" y="{y}" width="{x2 - x + 1}" height="1" fill="{PAL[c]}"/>')
            x = x2 + 1
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" shape-rendering="crispEdges">'
            f"<title>{title}</title>" + "".join(rects) + "</svg>")


# 5x7 pixel font (only the letters needed)
FONT = {
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "I": ["111", "010", "010", "010", "010", "010", "111"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
}


def wordmark(text, color="o"):
    cols = []
    for ch in text:
        gl = FONT[ch]
        for x in range(len(gl[0])):
            cols.append([color if gl[y][x] == "1" else "." for y in range(7)])
        cols.append(["."] * 7)
    return np.array(cols[:-1], dtype="<U1").T


def banner(pad=3):
    """Glass + wordmark on white with a margin of `pad` pixels."""
    big = np.repeat(np.repeat(wordmark("VOLLMILCH"), 2, 0), 2, 1)  # wordmark at 2x pixel size
    W = N + 3 + big.shape[1] + 1
    b = np.full((N, W), ".", dtype="<U1")
    b[:, :N] = g
    y0 = (N - big.shape[0]) // 2 + 2
    b[y0:y0 + big.shape[0], N + 3:N + 3 + big.shape[1]] = big
    b[b == "."] = "_"
    return np.pad(b, pad, constant_values="_")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    print("\n".join("".join(r) for r in g))
    open(f"{out}/vollmilch_logo.svg", "w").write(to_svg(g, "Vollmilch"))
    for s in (1, 2, 4, 8, 16):
        to_img(g, s).save(f"{out}/vollmilch_logo_{N * s}.png")
    bn = banner()
    open(f"{out}/vollmilch_banner.svg", "w").write(to_svg(bn, "Vollmilch"))
    to_img(bn, 6).save(f"{out}/vollmilch_banner.png")
