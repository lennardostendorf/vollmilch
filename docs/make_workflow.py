"""Vollmilch workflow diagram (SVG). python make_workflow.py docs/"""
import html
import re
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
LOGO = (OUT / "vollmilch_logo.svg").read_text() if (OUT / "vollmilch_logo.svg").exists() else ""

W, H = 1480, 380
INK, MUTED, LINE = "#1f2a3a", "#5c6b80", "#4a5a70"
FONT = "font-family=\"Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif\""
MONO = "font-family=\"'JetBrains Mono', Menlo, Consolas, monospace\""

STAGES = [
    dict(title="Inputs", cmd="vollmilch init · inspect", color="#5b7ba3", tint="#eef3f9",
         lines=["Multiplex image", "(qptiff, PNG, OME-TIFF)", "Regions from QuPath"]),
    dict(title="Segment", cmd="vollmilch segment", color="#2f8f83", tint="#ecf7f5",
         lines=["Cellpose-SAM segmentation", "Feature extraction"]),
    dict(title="Embed", cmd="vollmilch embed", color="#7a5ea8", tint="#f3effa",
         lines=["UMAP", "Leiden clustering", "Harmony integration"]),
    dict(title="Report & gate", cmd="vollmilch report", color="#c2832b", tint="#fbf4ea",
         lines=["Interactive HTML report", "Gating", "Gates export"]),
    dict(title="Export", cmd="vollmilch export", color="#4f8a3c", tint="#eff7ec",
         lines=["Populations to QuPath"]),
]

M, GAP = 36, 40
CW = (W - 2 * M - GAP * (len(STAGES) - 1)) / len(STAGES)
TOP, HEAD, BOX_H = 138, 58, 196

parts = []
a = parts.append


def text(x, y, s, size=13, weight=400, fill=INK, anchor="start", font=FONT, extra=""):
    a(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
      f'text-anchor="{anchor}" {font} {extra}>{html.escape(s)}</text>')


a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
a("""<defs>
<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#4a5a70"/></marker>
<filter id="sh" x="-5%" y="-5%" width="110%" height="120%"><feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#1f2a3a" flood-opacity="0.10"/></filter>
</defs>""")
a(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

if LOGO:
    inner = re.sub(r"^<svg[^>]*>|</svg>$", "", LOGO.strip())
    inner = re.sub(r"<title>.*?</title>", "", inner)
    a(f'<g transform="translate({M},26) scale(2)" shape-rendering="crispEdges">{inner}</g>')
text(M + 80, 62, "Vollmilch workflow", 28, 700)
text(M + 80, 90, "multiplex histology → cells → gates → QuPath", 15, 400, MUTED)

xs = [M + i * (CW + GAP) for i in range(len(STAGES))]
for i, (st, x) in enumerate(zip(STAGES, xs)):
    a(f'<g filter="url(#sh)"><rect x="{x}" y="{TOP}" width="{CW:.1f}" height="{BOX_H}" rx="12" '
      f'fill="{st["tint"]}" stroke="{st["color"]}" stroke-width="1.5"/></g>')
    a(f'<path d="M{x},{TOP + 12} a12,12 0 0 1 12,-12 h{CW - 24:.1f} a12,12 0 0 1 12,12 v{HEAD - 12} h{-CW:.1f} z" fill="{st["color"]}"/>')
    text(x + 16, TOP + 25, f"{i + 1}  {st['title']}", 17, 700, "#ffffff")
    text(x + 16, TOP + 45, st["cmd"], 12.5, 500, "#ffffff", font=MONO, extra='opacity="0.92"')
    # body lines, vertically centred in the body
    body_top, body_h, lh = TOP + HEAD, BOX_H - HEAD, 27
    y0 = body_top + body_h / 2 - (len(st["lines"]) - 1) * lh / 2 + 6
    for k, ln in enumerate(st["lines"]):
        sub = ln.startswith("(")
        text(x + CW / 2, y0 + k * lh - (5 if sub else 0), ln, 14 if sub else 16, 400 if sub else 500,
             MUTED if sub else INK, "middle")
    if i < len(STAGES) - 1:
        ay = TOP + HEAD + (BOX_H - HEAD) / 2
        a(f'<line x1="{x + CW + 4:.1f}" y1="{ay}" x2="{x + CW + GAP - 4:.1f}" y2="{ay}" '
          f'stroke="{LINE}" stroke-width="2.4" marker-end="url(#arr)"/>')
a("</svg>")
(OUT / "vollmilch_workflow.svg").write_text("\n".join(parts), encoding="utf-8")
print("written", OUT / "vollmilch_workflow.svg")
