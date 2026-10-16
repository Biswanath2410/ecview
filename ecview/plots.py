# plotly figures
import numpy as np
import plotly.graph_objects as go

from . import calc

COLS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
        "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD", "#1B9E77", "#E7298A"]


def runs(idx):
    # split sorted bins into contiguous runs
    out, cur = [], [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            out.append(cur)
            cur = [i]
    out.append(cur)
    return out


def label(b, i, res):
    r = b.loc[i]
    return f"bin {i}<br>{r.chrom}:{r.pos / 1e6:.3f}-{(r.pos + res) / 1e6:.3f} Mb ({r.ori})<br>segment {r.seg + 1}"


def lines(xyz, segs):
    # x, y, z lists for line pieces separated by None
    x, y, z = [], [], []
    for r in segs:
        x += list(xyz[r, 0]) + [None]
        y += list(xyz[r, 1]) + [None]
        z += list(xyz[r, 2]) + [None]
    return x, y, z


def flat(segs):
    # bins behind lines(), to redraw for another structure
    return sum([[int(i) for i in r] + [None] for r in segs], [])


def empty3d(**kw):
    return go.Scatter3d(x=[], y=[], z=[], hoverinfo="skip", showlegend=False, **kw)


def fig3d(xyz, b, cp, cons, genes, res, linear, others, lnc=()):
    f = go.Figure()
    segs = [g.index.tolist() for _, g in b.groupby("seg")]
    x, y, z = lines(xyz, segs)
    col = sum([[COLS[k % len(COLS)]] * len(r) + ["#000"] for k, r in enumerate(segs)], [])
    idx = sum([r + [None] for r in segs], [])
    f.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines+markers", name="ecDNA",
                             line=dict(color=col, width=6), marker=dict(size=2.5, color=col),
                             customdata=idx, hoverinfo="text", meta=idx,
                             hovertext=[label(b, i, res) if i is not None else "" for i in idx]))
    jn = [list(p) for p in calc.breaks(b, res, linear)]
    x, y, z = lines(xyz, jn)
    f.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="#666", width=4, dash="dash"),
                             hoverinfo="skip", name="junctions", meta=flat(jn)))
    pc = [[i, j] for i, c in enumerate(cp) for j in c if j > i]
    x, y, z = lines(xyz, pc)
    f.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="#999", width=1.5, dash="dot"),
                             hoverinfo="skip", name="copies", visible="legendonly", meta=flat(pc)))
    # contacts: one line per block, wider for more
    for name, df in cons.items():
        bl = con_blocks(df, len(b))
        first = True
        for lo, hi, w, _, op in SIZES:
            sel = [(m, t) for m, k, t in bl if lo <= k < hi]
            if not sel:
                continue
            pr = [m for m, _ in sel]
            x, y, z = lines(xyz, pr)
            f.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(width=w, color="#1f4e9c"), opacity=op + 0.15,
                                     name=name, legendgroup=name, showlegend=first,
                                     customdata=sum([[p, p, None] for p in pr], []), meta=flat(pr),
                                     hovertext=sum([[t, t, ""] for _, t in sel], []), hoverinfo="text",
                                     visible="legendonly"))  # off at first, as on the path panel
            first = False
    # lncRNA genes in their own layer, off at first
    for name, col, vis in [("genes", "crimson", True), ("lncRNA genes", "#888", "legendonly")]:
        sub = {k: v for k, v in genes.items() if (k in lnc) == (name != "genes")}
        if not sub:
            continue
        rr, txt, cd, ht = [], [], [], []
        for gn, gi in sub.items():
            for r in runs(gi):
                rr.append(r)
                txt += [gn if i == r[len(r) // 2] else "" for i in r] + [""]
                cd += r + [None]
                ht += [gn] * len(r) + [""]
        x, y, z = lines(xyz, rr)
        f.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines+text", text=txt, name=name, visible=vis,
                                 textfont=dict(size=11, color=col), line=dict(color=col, width=9),
                                 opacity=0.5, customdata=cd, hovertext=ht, hoverinfo="text", meta=cd))
    for name, o in others.items():
        x, y, z = lines(o, segs)
        f.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="#777", width=3), opacity=0.6,
                                 hoverinfo="skip", name=name, visible="legendonly"))
    # filled by the page: neighbors, picks, copies
    f.add_trace(empty3d(mode="markers", marker=dict(size=5, color="gold", line=dict(color="black", width=1))))
    # picked = black, copies = orange
    f.add_trace(empty3d(mode="lines", line=dict(color="black", width=5)))
    f.add_trace(empty3d(mode="markers", marker=dict(size=9, color="black", line=dict(color="white", width=1.5))))
    f.add_trace(empty3d(mode="markers", marker=dict(size=7, color="darkorange", symbol="diamond")))
    ax = dict(visible=False)
    f.update_layout(scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="data", domain=dict(y=[0, 0.84])),
                    scene_dragmode="orbit",
                    margin=dict(l=0, r=0, t=30, b=0, autoexpand=False), title="3D structure", uirevision="keep",
                    legend=dict(orientation="h", x=0, y=1, yanchor="top", font=dict(size=12),
                                bgcolor="rgba(255,255,255,0.8)", title=dict(text="show/hide:"), itemdoubleclick=False))
    return f


def heat(z, hov):
    return go.Heatmap(z=z, colorscale="Reds", zmin=0, zmax=float(np.percentile(z, 99.5)), hovertemplate=hov,
                      colorbar=dict(title="log1p", len=0.6))


def square(f, title, shp, ticks=None):
    f.update_layout(shapes=shp, title=title, margin=dict(l=60, r=10, t=30, b=40), showlegend=False,
                    yaxis=dict(autorange="reversed", scaleanchor="x", constrain="domain"),
                    xaxis=dict(constrain="domain"), plot_bgcolor="white", uirevision="keep")
    if ticks:
        f.update_xaxes(tickvals=ticks[0], ticktext=ticks[1], tickangle=-45, tickfont=dict(size=9))
        f.update_yaxes(tickvals=ticks[0], ticktext=ticks[1], tickfont=dict(size=9))
    return f


def cuts(at, n):
    shp = []
    for e in at:
        shp += [dict(type="line", x0=e - 0.5, x1=e - 0.5, y0=-0.5, y1=n - 0.5, line=dict(width=0.6, color="#555")),
                dict(type="line", y0=e - 0.5, y1=e - 0.5, x0=-0.5, x1=n - 0.5, line=dict(width=0.6, color="#555"))]
    return shp


def figmap(m, b, cons):
    f = go.Figure(heat(np.log1p(m), "bins %{y}, %{x}<br>log1p %{z:.2f}<extra></extra>"))
    for name, df in cons.items():
        lo, hi = np.minimum(df.bin1, df.bin2), np.maximum(df.bin1, df.bin2)
        f.add_trace(go.Scatter(x=hi, y=lo, mode="markers", opacity=0.7,
                               marker=dict(size=3, symbol="circle-open", color="#1f4e9c"), name=name,
                               hovertemplate="bins %{y}, %{x}<extra>" + name + "</extra>"))
    at = b.index[b.seg != b.seg.shift()].tolist()[1:]
    return square(f, "Hi-C along the ecDNA (copies separate)", cuts(at, len(b)))


def figref(c, keys, res):
    # genome order, copies summed
    f = go.Figure(heat(np.log1p(c), "reference bins %{y}, %{x}<br>log1p %{z:.2f}<extra></extra>"))
    at = [r for r in range(1, len(keys)) if keys[r][0] != keys[r - 1][0] or keys[r][1] - keys[r - 1][1] != res]
    starts = [0]
    for r in at:
        if r - starts[-1] > len(keys) / 25:  # keep labels apart
            starts.append(r)
    tick = (starts, [f"{keys[r][0]} {keys[r][1] / 1e6:.2f}" for r in starts])
    return square(f, "Hi-C in genome order (copies summed)", cuts(at, len(keys)), tick)


def arcs(pairs, n, d):
    # arcs under the path, deeper for pairs further apart
    x, y, cd = [], [], []
    t = np.linspace(0, np.pi, 20)
    for i, j in pairs:
        x += list(i + (j - i) * (1 - np.cos(t)) / 2) + [None]
        y += list(-0.1 - d * np.sin(t) * np.sqrt((j - i) / n)) + [None]
        cd += [[i, j]] * 20 + [None]
    return x, y, cd


def blocks(pairs, n, parts=30):
    # contacts grouped by the path windows of their ends
    w = max(3, round(n / parts))
    out = {}
    for i, j in pairs:
        out.setdefault((i // w, j // w), []).append((i, j))
    return list(out.values())


TOP = 2.1  # room above the bar for segment labels

# block size -> width in 3D, width and opacity on the path
SIZES = [(1, 3, 1, 0.5, 0.25), (3, 10, 2, 1, 0.45), (10, 30, 3.5, 2, 0.6), (30, 10 ** 9, 5, 3.5, 0.7)]


def con_blocks(df, n):
    # -> [(middle contact, count, text)] per block
    pairs = sorted({(min(i, j), max(i, j)) for i, j in zip(df.bin1, df.bin2) if i != j})
    out = []
    for bl in blocks(pairs, n):
        a, c = [p[0] for p in bl], [p[1] for p in bl]
        mi, mj = np.median(a), np.median(c)
        mid = min(bl, key=lambda p: abs(p[0] - mi) + abs(p[1] - mj))
        out.append((list(mid), len(bl), f"bins {min(a)}-{max(a)} with {min(c)}-{max(c)}: "
                    f"{len(bl)} contact{'s' * (len(bl) > 1)}"))
    return out


def figpath(b, cp, genes, res, tracks, lnc=(), cons=None):
    n = len(b)
    f = go.Figure()
    shp, ann = [], []
    for k, grp in b.groupby("seg"):
        a, e = grp.index[0], grp.index[-1]
        shp.append(dict(type="rect", x0=a - 0.5, x1=e + 0.5, y0=0, y1=0.35, fillcolor=COLS[k % len(COLS)],
                        line=dict(width=0.5, color="white")))
        s = grp.pos.min() / 1e6
        ann.append(dict(x=(a + e) / 2, y=0.4, text=f"{grp.chrom.iloc[0]} {s:.2f} {grp.ori.iloc[0]}",
                        showarrow=False, textangle=-40, font=dict(size=10), yanchor="bottom"))
    # arc layers ("arc:..."), each with a checkbox
    con = {k: con_blocks(df, n) for k, df in (cons or {}).items()}
    d = 2.2 if any(con.values()) else 0.9
    # one arc per contact block; a hidden dot on top for hover and click
    for name, bl in con.items():
        if not bl:
            continue
        for lo, hi, _, w, op in SIZES:
            sel = [m for m, k, _ in bl if lo <= k < hi]
            if sel:
                ax, ay, _ = arcs(sel, n, d)
                f.add_trace(go.Scatter(x=ax, y=ay, mode="lines", line=dict(color="#1f4e9c", width=w), opacity=op,
                                       hoverinfo="skip", name="arc:" + name, visible=False))
        top = [arcs([m], n, d) for m, _, _ in bl]
        f.add_trace(go.Scatter(x=[t[0][10] for t in top], y=[t[1][10] for t in top], mode="markers",
                               marker=dict(size=10, opacity=0), customdata=[m for m, _, _ in bl],
                               hovertext=[t for _, _, t in bl], hoverinfo="text", name="arc:" + name, visible=False))
    pairs = [(i, j) for i, c in enumerate(cp) for j in c if j > i]
    if pairs:  # drawn last, so it stays visible over contact arcs
        ax, ay, _ = arcs(pairs, n, d)
        f.add_trace(go.Scatter(x=ax, y=ay, mode="lines", line=dict(color="#777", width=0.8), hoverinfo="skip",
                               name="arc:duplicated regions"))
    f.add_trace(go.Scatter(x=list(range(n)), y=[0.175] * n, mode="markers",
                           marker=dict(size=6, color=["black" if cp[i] else "rgba(0,0,0,0)" for i in range(n)], symbol="line-ns-open"),
                           customdata=list(range(n)), hovertext=[label(b, i, res) for i in range(n)],
                           hoverinfo="text"))
    for gn, idx in genes.items():
        col = "#999" if gn in lnc else "crimson"
        for r in runs(idx):
            shp.append(dict(type="rect", x0=r[0] - 0.5, x1=r[-1] + 0.5, y0=-0.35 - d, y1=-0.15 - d, fillcolor=col,
                            line=dict(width=0)))
            ann.append(dict(x=r[len(r) // 2], y=-0.4 - d, text=gn, showarrow=False, textangle=-60,
                            font=dict(size=9, color=col), yanchor="top"))
    y0 = -1.6 - d
    for k, (name, v) in enumerate(tracks.items()):
        top = y0 - 0.1 - k * 0.8
        h = v / np.nanmax(v) * 0.6 if np.nanmax(v) > 0 else v
        f.add_trace(go.Scatter(x=list(range(n)), y=top - 0.6 + h, mode="lines", line=dict(width=1, color="#333"), customdata=list(range(n)), hoverinfo="text", meta=name,
                               hovertext=[f"{name}: {x:.3g}<br>bin {i}" for i, x in enumerate(v)]))
        # name="trk": moved by the page
        ann.append(dict(x=-1, y=top - 0.3, text=name, showarrow=False, xanchor="right", font=dict(size=10), name="trk"))
        shp.append(dict(type="line", x0=-0.5, x1=n - 0.5, y0=top - 0.6, y1=top - 0.6, line=dict(width=0.5, color="#bbb"),
                        name="trk"))
    low = y0 - 0.1 - len(tracks) * 0.8 if tracks else y0 + 0.1
    f.update_layout(shapes=shp, annotations=ann, showlegend=False, plot_bgcolor="white", xaxis=dict(range=[-1, n], title="path bin", zeroline=False),
                    yaxis=dict(visible=False, range=[low, TOP]), margin=dict(l=max([70] + [6 * len(t) + 20 for t in tracks]), r=60, t=10, b=40),
                    hovermode="closest", uirevision="keep", meta=dict(y0=y0, top=TOP, h=230 + round((d - 0.9) * 36)))
    return f
