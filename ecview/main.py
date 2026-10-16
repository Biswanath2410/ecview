#!/usr/bin/env python3
# ecView: linked 3D, Hi-C and path views of an ecDNA
import argparse
import base64
import json
import os

import numpy as np
import pandas as pd
import plotly.io as pio
from plotly.offline import get_plotlyjs, get_plotlyjs_version

from . import calc, deconv, files, plots, structs

HERE = os.path.dirname(os.path.abspath(__file__))


def named(items):
    # "name=file" or "file" -> {name: file}
    items = list(items or [])
    stem = [os.path.basename(x).split(".")[0] for x in items if "=" not in x]
    pre = os.path.commonprefix(stem) if len(stem) > 1 else ""
    suf = os.path.commonprefix([t[::-1] for t in stem])[::-1] if len(stem) > 1 else ""
    pre = pre[:pre.rfind("_") + 1]
    suf = suf[suf.find("_"):] if "_" in suf else ""
    out = {}
    for x in items:
        if "=" in x:
            k, f = x.split("=", 1)
        else:
            f = x
            k = os.path.basename(x).split(".")[0]
            k = k[len(pre):len(k) - len(suf)] or k
        out[k] = f
    return out


def pick_bins(bed, res, x, m, log):
    # ec3D or CADET binning, whichever fits the inputs
    x = x if x is None or isinstance(x, int) else len(x)
    nb, nr = {}, {}
    for h in ("ec3d", "cadet"):
        bh = files.path_bins(bed, res, h)
        nb[h], nr[h] = len(bh), len(calc.ref_index(bh)[1])
    if not max(nb.values()):
        raise SystemExit(f"the path has no {res} bp bins (its segments are shorter than a bin); "
                         "pick another cycle or a smaller --res")
    ok = [h for h in nb if (x is not None or m is not None)
          and (m is None or len(m) == nb[h]) and (x is None or x in (nb[h], nr[h]))]
    got = {v for v in (x, None if m is None else len(m)) if v is not None}
    if got and not ok:
        raise SystemExit(f"inputs have {sorted(got)} bins, the path has {nb['ec3d']} (ec3D binning) "
                         f"or {nb['cadet']} (CADET binning)")
    if got and len(ok) == 2:
        log.append("ec3D and CADET binning give the same number of bins; using ec3D (--bins cadet to change)")
    return "ec3d" if not ok or "ec3d" in ok else "cadet"


def plain(o):
    # plotly 6+ stores arrays as base64; the page needs lists
    if isinstance(o, dict):
        if "bdata" in o and "dtype" in o:
            a = np.frombuffer(base64.b64decode(o["bdata"]), dtype=o["dtype"])
            if "shape" in o:
                a = a.reshape([int(n) for n in str(o["shape"]).split(",")])
            return np.where(np.isnan(a), None, a).tolist() if a.dtype.kind == "f" else a.tolist()
        return {k: plain(v) for k, v in o.items()}
    if isinstance(o, list):
        return [plain(v) for v in o]
    return o


def esc(s):
    # safe inside <script>, as plotly does
    return s.replace("<", "\\u003c").replace(">", "\\u003e").replace("/", "\\u002f")


def fjson(f):
    return esc(json.dumps(plain(json.loads(pio.to_json(f))), separators=(",", ":"))) if f else "null"


def build(bed=None, cycles=None, cycle=1, xyz=None, res=5000, mat=None, norm="NONE", contacts=(),
          genes=None, feat=(), track=(), linear=False, cdn=False, title="ecview", deconvolve=False, cadet=None,
          bins="auto"):
    if cycles:
        bed, lin = files.read_cycles(cycles, cycle)
        linear = linear or lin
    else:
        bed = files.read_bed(bed)
    log, m, pi = [], None, None
    xs = [xyz] if isinstance(xyz, str) else list(xyz or [])  # first structure is the reference
    xyz, more = (xs[0] if xs else None), xs[1:]
    rn, xyz = next(iter(named([xyz]).items())) if xyz else (None, None)
    sx = structs.read(xyz) if xyz else None
    hic = bool(mat) and (mat.endswith((".hic", ".cool", ".mcool")) or "::" in mat)
    if mat and not hic:
        m = files.load_mat(mat, None, res)
    if deconvolve:
        if not hic:
            raise SystemExit("--deconvolve needs --mat as .hic, .cool or .mcool")
        b, m, pi, _ = deconv.run(deconv.load(cadet), bed, mat, res, norm, log.append)
        how = "cadet"
    else:
        nx = len(sx[1]) if sx and sx[0] == "rows" else None
        how = bins if bins in ("ec3d", "cadet") else pick_bins(bed, res, nx, m, log)
        b = files.path_bins(bed, res, how)
    n = len(b)
    if n == 0:
        raise SystemExit(f"the path has no {res} bp bins; pick another cycle or a smaller --res")
    cp = calc.copies(b)
    ref, keys = calc.ref_index(b)
    log.insert(0, f"{n} bins ({how} binning), {len(keys)} reference bins, {sum(1 for c in cp if c)} duplicated")
    x = None
    if sx:
        x, note = structs.place(*sx, b, res, ref, len(keys))
        if note:
            log.append(f"{rn}: {note}")
    if hic and not deconvolve:
        m = files.load_mat(mat, b, res, norm)
        if any(cp):
            log.append("contacts of duplicated regions split evenly between copies (not deconvolved)")
    if m is not None and m.shape != (n, n):
        raise SystemExit(f"{mat}: matrix is {m.shape}, expected {n} x {n} ({how} binning)")
    cons = {}
    for k, fn in named(contacts).items():
        cons[k] = files.load_contacts(fn)
    gn = {k: v for k, v in files.load_feats(genes, b, res)} if genes else {}
    lnc = {k for k, t in (files.gene_types(genes) if genes else {}).items() if t != "protein_coding" and k in gn}
    fts = {"gene": [[k, v] for k, v in gn.items()]}
    for k, fn in named(feat).items():
        fts[k] = files.load_feats(fn, b, res)
    trk = {k: files.load_track(fn, b, res) for k, fn in named(track).items()}
    for k, fs in fts.items():
        if k != "gene":  # features also as a track: how many in each bin
            trk[k] = np.bincount([i for _, idx in fs for i in idx], minlength=n).astype(float)
    if pi is not None:
        trk["copy weight (pi)"] = np.where([bool(c) for c in cp], pi, np.nan)

    colors, others, rmsd, tot = {}, {}, {}, None
    if x is not None and m is not None:
        colors["fit to Hi-C"], tot = calc.fit(x, m, linear)
        log.append(f"structure vs Hi-C: Spearman {tot:.3f}")
    if x is not None:
        for k, fn in named(more).items():
            while k == rn or k in others:  # names must differ
                k += "'"
            o, note = structs.place(*structs.read(fn), b, res, ref, len(keys))
            if note:
                log.append(f"{k}: {note}")
            y, r = calc.align(o, x)
            if r < 1e-6 * calc.spacing(x):
                log.append(f"{k}: same as --xyz, skipped")
                continue
            others[k], rmsd[k] = y, r
            log.append(f"{k}: RMSD {r / calc.spacing(x):.2f} bin spacings after alignment")
        if len(others) == 1:
            k = list(others)[0]
            colors[f"shift vs {k}"] = np.linalg.norm(others[k] - x, axis=1) / calc.spacing(x)
        elif others:
            colors["spread"] = calc.spread([x] + list(others.values())) / calc.spacing(x)
    # every structure aligned on the reference
    rnd = lambda v: [None if np.isnan(a) else round(float(a), 4) for a in v]
    st, shift = [], {}
    if x is not None:
        allx = {rn: x, **others}
        for k, v in allx.items():
            fk, tk = (colors["fit to Hi-C"], tot) if k == rn and m is not None else \
                calc.fit(v, m, linear) if m is not None else (np.full(n, np.nan), None)
            st.append(dict(name=k, xyz=np.round(v, 5).tolist(), step=calc.spacing(v), fit=rnd(fk), tot=tk,
                           rmsd=0.0 if k == rn else rmsd[k] / calc.spacing(x),
                           cov=int((~np.isnan(v[:, 0])).sum())))
        for a in allx:  # per-bin shift of every other structure, aligned on this one
            shift[a] = {c: rnd(np.linalg.norm(calc.align(allx[c], allx[a])[0] - allx[a], axis=1)
                               / calc.spacing(allx[a])) for c in allx if c != a}
    nc = np.array([len(c) + 1 for c in cp])
    for t, v in trk.items():
        colors[t] = v
        if t in named(track) and (nc > 1).any():
            colors[f"{t} per copy"] = v / nc  # signal of a repeated region is shared by its copies

    f3 = plots.fig3d(x, b, cp, cons, gn, res, linear, others, lnc) if x is not None else None
    fm = fr = None
    if m is not None:
        fm = plots.figmap(np.round(m, 3), b, cons)
        fr = plots.figref(np.round(calc.collapse(m, ref, len(keys)), 3), keys, res)
    fp = plots.figpath(b, cp, gn, res, trk, lnc, cons)

    con = {f"{min(i, j)},{max(i, j)}": [k, None if pd.isna(s) else float(s)]
           for k, df in cons.items() for i, j, s in zip(df.bin1, df.bin2, df.sc)}
    data = dict(chrom=b.chrom.tolist(), pos=b.pos.tolist(), ori=b.ori.tolist(), seg=(b.seg + 1).tolist(),
                xyz=np.round(x, 5).tolist() if x is not None else None, cp=cp, res=res, linear=linear,
                con=con, genes=gn, ref=ref.tolist(), rk=[[c, int(p)] for c, p in keys], feats=fts,
                colors={k: [None if np.isnan(v) else round(float(v), 4) for v in c] for k, c in colors.items()},
                step=calc.spacing(x) if x is not None else 1, fit=tot,
                rmsd={k: v / calc.spacing(x) for k, v in rmsd.items()}, structs=st, shift=shift)
    js = f'<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>' if cdn else \
        f"<script>{get_plotlyjs()}</script>"
    html = open(os.path.join(HERE, "template.html")).read()
    for k, v in [("{{PLOTLY}}", js), ("{{TITLE}}", title), ("{{DATA}}", esc(json.dumps(data))),
                 ("{{F3}}", fjson(f3)), ("{{FM}}", fjson(fm)), ("{{FR}}", fjson(fr)), ("{{FP}}", fjson(fp))]:
        html = html.replace(k, v)
    return html, log


def main():
    ap = argparse.ArgumentParser(description="Linked 3D / Hi-C / path view of an ecDNA")
    ap.add_argument("--bed", help="ecDNA path BED (chrom start end orientation)")
    ap.add_argument("--cycles", help="AmpliconArchitect or CoRAL cycles file, instead of --bed")
    ap.add_argument("--cycle", type=int, default=1, help="cycle number in --cycles")
    ap.add_argument("--xyz", nargs="*", default=[], help="3D structures ([name=]file); the first is the reference")
    ap.add_argument("--res", type=int, default=5000, help="bin size (bp)")
    ap.add_argument("--bins", default="auto", choices=["auto", "ec3d", "cadet"], help="how the path is binned")
    ap.add_argument("--mat", help="Hi-C along the ecDNA: .txt, .npy, .hic, .cool or .mcool")
    ap.add_argument("--norm", default="NONE", help="normalization for .hic/.cool (e.g. KR)")
    ap.add_argument("--deconvolve", action="store_true", help="split copies with CADET Layer 1")
    ap.add_argument("--cadet", help="CADET folder (for --deconvolve)")
    ap.add_argument("--contacts", nargs="*", default=[], help="tables with bin1, bin2 ([name=]file)")
    ap.add_argument("--genes", help="gene BED (chrom start end name [strand type])")
    ap.add_argument("--feat", nargs="*", default=[], help="other features, BED ([name=]file)")
    ap.add_argument("--track", nargs="*", default=[], help="bedGraph, bigWig or BED ([name=]file)")
    ap.add_argument("--linear", action="store_true", help="do not close the circle")
    ap.add_argument("--cdn", action="store_true", help="load plotly.js from the web")
    ap.add_argument("-o", "--out", default="ecview.html")
    a = ap.parse_args()
    if not a.bed and not a.cycles:
        ap.error("give --bed or --cycles")
    if not a.xyz and not a.mat:
        ap.error("give --xyz, --mat or both")
    html, log = build(a.bed, a.cycles, a.cycle, a.xyz, a.res, a.mat, a.norm, a.contacts, a.genes,
                      a.feat, a.track, a.linear, a.cdn, os.path.basename(a.out).rsplit(".", 1)[0],
                      a.deconvolve, a.cadet, a.bins)
    open(a.out, "w").write(html)
    print("\n".join(log) + f"\n-> {a.out}")


if __name__ == "__main__":
    main()
