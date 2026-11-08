# ruff: noqa: E501  (SVG and CSS templates are easier to read unwrapped)
"""Charts for the static site: small, dependency-free inline SVG line charts.

Adapted from the sister projects' site_charts.py (same CSS custom properties, tooltip and table view),
with a generic x/y line chart in place of their catch-curve chart.

Colors are CSS custom properties (--s1.. series, --ink.. chrome) defined once in the page, so the light
and dark steps swap in one place. Series palette: blue, orange, aqua from the documented categorical
order, validated against this page's light (#fbfbfa) and dark (#151513) surfaces. Aqua is below 3:1 on
light, so every line carries a direct label where it fits, a legend, and every chart has a table view.
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable

import numpy as np

CHART_CSS = """
:root { --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --grid:#e3e2dc; --axis:#c3c2b7; --ink-muted:#898781; --surface:#fbfbfa; }
@media (prefers-color-scheme: dark) { :root { --s1:#3987e5; --s2:#d95926; --s3:#199e70; --grid:#2c2c2a; --axis:#383835; --surface:#151513; } }
.chart { margin:12px 0 4px; overflow-x:auto; }
.chart svg { width:100%; min-width:330px; max-width:620px; height:auto; display:block; font:13px system-ui,-apple-system,"Segoe UI",sans-serif; }
.chart .grid line { stroke:var(--grid); stroke-width:1; }
.chart .axis { stroke:var(--axis); stroke-width:1; }
.chart .tick { fill:var(--ink-muted); font-variant-numeric:tabular-nums; }
.chart .lbl { fill:var(--fg); }
.chart .muted { fill:var(--ink-muted); }
.chart .ref { stroke:var(--ink-muted); stroke-width:1; stroke-dasharray:4 3; }
.chart .series { fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }
.chart .dot { stroke:var(--surface); stroke-width:2; }
.chart .hit { fill:transparent; cursor:crosshair; }
.chart .hit:focus { outline:none; }
.chart .xhair { stroke:var(--ink-muted); stroke-width:1; visibility:hidden; }
.legend { display:flex; flex-wrap:wrap; gap:6px 16px; font-size:.85rem; color:var(--muted); margin:4px 0; }
.legend span::before { content:""; display:inline-block; width:14px; height:2px; vertical-align:middle; margin-right:6px; background:var(--k); }
#tip { position:fixed; pointer-events:none; z-index:10; background:var(--bg); color:var(--fg); border:1px solid var(--line);
  border-radius:6px; padding:6px 9px; font-size:.82rem; line-height:1.4; box-shadow:0 2px 8px rgba(0,0,0,.12); display:none; max-width:260px; }
#tip .row { display:flex; align-items:center; gap:6px; white-space:nowrap; }
#tip .row i { display:inline-block; width:12px; height:2px; }
#tip b { font-variant-numeric:tabular-nums; }
details.tableview { margin:4px 0 8px; font-size:.88rem; }
details.tableview table { min-width:0; width:auto; }
"""

# One tooltip for the whole page. Values come from JSON in data-* attributes and are inserted with
# textContent only. Each chart gets a crosshair that snaps to the nearest x; the hit area is focusable
# and shows the same readout from the keyboard.
CHART_JS = """
(() => {
  const tip = document.getElementById('tip');
  const show = (ev, rows, title) => {
    tip.replaceChildren();
    if (title) { const t = document.createElement('div'); t.textContent = title; t.style.color = 'var(--muted)'; tip.append(t); }
    for (const [color, value, name] of rows) {
      const r = document.createElement('div'); r.className = 'row';
      if (color) { const k = document.createElement('i'); k.style.background = color; r.append(k); }
      const b = document.createElement('b'); b.textContent = value; r.append(b);
      const s = document.createElement('span'); s.textContent = name; r.append(s);
      tip.append(r);
    }
    tip.style.display = 'block';
    const box = ev.target.getBoundingClientRect();
    const x = ev.clientX ?? (box.left + box.width / 2), y = ev.clientY ?? box.top;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.max(8, Math.min(window.innerWidth - w - 8, x + 14)) + 'px';
    tip.style.top = Math.max(8, y - h - 10) + 'px';
  };
  const hide = () => { tip.style.display = 'none'; };

  for (const svg of document.querySelectorAll('svg[data-xy]')) {
    const d = JSON.parse(svg.dataset.xy);
    const hit = svg.querySelector('.hit'), xh = svg.querySelector('.xhair');
    const at = (clientX) => {
      const pt = svg.createSVGPoint(); pt.x = clientX; pt.y = 0;
      const p = pt.matrixTransform(svg.getScreenCTM().inverse());
      const fx = d.xmin + (p.x - d.x0) / (d.x1 - d.x0) * (d.xmax - d.xmin);
      let i = 0, best = Infinity;
      d.grid.forEach((g, k) => { const e = Math.abs(g - fx); if (e < best) { best = e; i = k; } });
      return i;
    };
    const render = (ev, i) => {
      const x = d.x0 + (d.grid[i] - d.xmin) / (d.xmax - d.xmin) * (d.x1 - d.x0);
      xh.setAttribute('x1', x); xh.setAttribute('x2', x); xh.style.visibility = 'visible';
      show(ev, d.series.map(s => [s.color, s.y[i].toFixed(d.dec) + d.unit, s.name]), d.labels[i]);
    };
    hit.addEventListener('pointermove', ev => render(ev, at(ev.clientX)));
    hit.addEventListener('pointerleave', () => { hide(); xh.style.visibility = 'hidden'; });
    hit.addEventListener('focus', ev => render(ev, Math.floor(d.grid.length / 2)));
    hit.addEventListener('blur', () => { hide(); xh.style.visibility = 'hidden'; });
  }
})();
"""


def xy_chart(
    chart_id: str,
    title: str,
    grid: np.ndarray,
    series: list[dict],
    *,
    xlabel: str,
    ylabel: str,
    xmin: float,
    xmax: float,
    ymax: float,
    xtick_fmt: Callable[[float], str],
    ytick_fmt: Callable[[float], str],
    labels: list[str],
    value_fmt: Callable[[float], str],
    table_x: list[float],
    table_head: list[str],
    table_row_label: str,
    ref_label: str | None = None,
    dec: int = 1,
    unit: str = "",
    xticks: int = 5,
    end_labels: bool = True,
) -> str:
    """Line chart on a shared x grid.

    series: [{"name", "short", "var" (css var name, e.g. "--s1"), "y" (array matching grid)}].
    labels: one tooltip title per grid point. table_x/table_head: the x values (and their column
    headings) shown in the "Show as a table" view; values come from the same arrays as the lines.
    ref_label: draws a dashed diagonal from (xmin, 0) to (xmax, ymax) with this label (chance line).
    """
    W, H, L, R, T, B = 500, 300, 46, 104, 14, 44
    x0, x1, y0, y1 = L, W - R, H - B, T
    grid = np.asarray(grid, float)
    X = lambda v: x0 + (v - xmin) / (xmax - xmin) * (x1 - x0)  # noqa: E731
    Y = lambda v: y0 - v / ymax * (y0 - y1)  # noqa: E731
    parts = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="{chart_id}-t" data-xy="__DATA__">',
        f'<title id="{chart_id}-t">{html.escape(title)}</title>',
        '<g class="grid">',
    ]
    for f in (0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(f * ymax):.1f}" y2="{Y(f * ymax):.1f}"/>')
    parts.append("</g>")
    for f in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<text class="tick" x="{x0 - 6}" y="{Y(f * ymax) + 4:.1f}" text-anchor="end">{html.escape(ytick_fmt(f * ymax))}</text>'
        )
    for v in np.linspace(xmin, xmax, xticks):
        parts.append(
            f'<text class="tick" x="{X(v):.1f}" y="{y0 + 16}" text-anchor="middle">{html.escape(xtick_fmt(float(v)))}</text>'
        )
    parts.append(f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}"/>')
    parts.append(
        f'<text class="muted" x="{(x0 + x1) / 2:.0f}" y="{H - 6}" text-anchor="middle">{html.escape(xlabel)}</text>'
    )
    parts.append(
        f'<text class="muted" transform="translate(11 {(y0 + y1) / 2:.0f}) rotate(-90)" text-anchor="middle">{html.escape(ylabel)}</text>'
    )
    if ref_label:
        parts.append(f'<line class="ref" x1="{X(xmin):.1f}" y1="{Y(0):.1f}" x2="{X(xmax):.1f}" y2="{Y(ymax):.1f}"/>')
        parts.append(f'<text class="muted" x="{X(xmax) - 4:.1f}" y="{Y(ymax) + 16:.1f}" text-anchor="end">{html.escape(ref_label)}</text>')
    # Direct end labels: keep them only where they don't collide (min 14px apart); the legend always carries identity.
    ends = sorted(((Y(float(np.asarray(s["y"])[-1])), s) for s in series), key=lambda t: t[0])
    labelled, last = set(), -1e9
    for ypos, s in ends:
        if end_labels and ypos - last >= 14:
            labelled.add(s["name"])
            last = ypos
    for s in series:
        ys = np.asarray(s["y"], float)
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(grid, ys, strict=True))
        parts.append(f'<polyline class="series" style="stroke:var({s["var"]})" points="{pts}"/>')
        parts.append(f'<circle class="dot" r="4" cx="{X(grid[-1]):.1f}" cy="{Y(ys[-1]):.1f}" style="fill:var({s["var"]})"/>')
        if s["name"] in labelled:
            parts.append(
                f'<text class="lbl" x="{X(grid[-1]) + 8:.1f}" y="{Y(ys[-1]) + 4:.1f}">{html.escape(value_fmt(float(ys[-1])))} {html.escape(s.get("short", s["name"]))}</text>'
            )
    parts.append(f'<line class="xhair" x1="0" x2="0" y1="{y1}" y2="{y0}"/>')
    parts.append(
        f'<rect class="hit" tabindex="0" x="{x0}" y="{y1}" width="{x1 - x0}" height="{y0 - y1}"><title>{html.escape(title)}</title></rect>'
    )
    parts.append("</svg>")
    data = {
        "x0": x0,
        "x1": x1,
        "xmin": float(xmin),
        "xmax": float(xmax),
        "dec": dec,
        "unit": unit,
        "labels": labels,
        "grid": [round(float(v), 5) for v in grid],
        "series": [
            {
                "name": s["name"],
                "color": f"var({s['var']})",
                "y": [round(float(v), 4) for v in np.asarray(s["y"], float)],
            }
            for s in series
        ],
    }
    svg = "".join(parts).replace("__DATA__", html.escape(json.dumps(data), quote=True))
    legend = (
        '<div class="legend">'
        + "".join(f'<span style="--k:var({s["var"]})">{html.escape(s["name"])}</span>' for s in series)
        + (f'<span style="--k:var(--ink-muted)">{html.escape(ref_label)}</span>' if ref_label else "")
        + "</div>"
    )
    rows = "".join(
        "<tr><td>"
        + html.escape(s["name"])
        + "</td>"
        + "".join(
            f'<td class="num">{html.escape(value_fmt(float(np.interp(p, grid, s["y"]))))}</td>' for p in table_x
        )
        + "</tr>"
        for s in series
    )
    table = (
        '<details class="tableview"><summary>Show as a table</summary><div class="wrap"><table><thead><tr>'
        f"<th>{html.escape(table_row_label)}</th>"
        + "".join(f'<th class="num">{html.escape(h)}</th>' for h in table_head)
        + f"</tr></thead><tbody>{rows}</tbody></table></div></details>"
    )
    return f'{legend}<div class="chart">{svg}</div>{table}'
