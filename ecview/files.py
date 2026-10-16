# reading inputs
import io
import os
import re

import numpy as np
import pandas as pd


def read_bed(fn):
    df = pd.read_csv(fn, sep=r"\s+", header=None, comment="#")
    if not str(df.iloc[0, 1]).isdigit():
        df = df.iloc[1:]
    # orientation in column 4, or column 6 as in ec3D cycle files
    o = 3 if df[3].astype(str).isin(["+", "-"]).all() else 5
    df = df.iloc[:, [0, 1, 2, o]]
    df.columns = ["chrom", "start", "end", "ori"]
    df[["start", "end"]] = df[["start", "end"]].astype(int)
    return df.reset_index(drop=True)


def read_cycles(fn, k=1):
    # AmpliconArchitect / CoRAL cycles file -> path BED
    seg, cyc = {}, {}
    for l in open(fn):
        f = l.rstrip("\n").split("\t")
        if f[0] == "Segment":
            seg[f[1]] = (f[2], int(f[3]), int(f[4]))
        elif l.startswith("Cycle="):
            d = dict(x.split("=", 1) for x in l.strip().split(";"))
            cyc[int(d["Cycle"])] = d["Segments"].split(",")
    if k not in cyc:
        raise SystemExit(f"cycle {k} not in {fn} (found {sorted(cyc)})")
    # 0+ / 0- mark the ends of a linear path
    lin = any(s.startswith("0") for s in cyc[k])
    rows = []
    for s in cyc[k]:
        if s.startswith("0"):
            continue
        c, a, e = seg[s[:-1]]
        rows.append([c, a, e, s[-1]])
    # merge segments that continue to each other
    out = [rows[0]]
    for r in rows[1:]:
        if joins(out[-1], r):
            out[-1] = merge(out[-1], r)
        else:
            out.append(r)
    if not lin and len(out) > 1 and joins(out[-1], out[0]):
        out = [merge(out[-1], out[0])] + out[1:-1]
    return pd.DataFrame(out, columns=["chrom", "start", "end", "ori"]), lin


def joins(a, b):
    if a[0] != b[0] or a[3] != b[3]:
        return False
    return b[1] == a[2] + 1 if a[3] == "+" else a[1] == b[2] + 1


def merge(a, b):
    return [a[0], min(a[1], b[1]), max(a[2], b[2]), a[3]]


def path_bins(bed, res, how="ec3d"):
    # ec3d rounds segment ends, cadet floors them
    rows = []
    for k, s in bed.iterrows():
        if how == "cadet":
            pos = list(range(s.start // res * res, s.end // res * res, res))
        else:
            pos = list(range(round(s.start / res) * res, round(s.end / res) * res, res))
        if s.ori == "-":
            pos = pos[::-1]
        rows += [(k, s.chrom, p, s.ori) for p in pos]
    df = pd.DataFrame(rows, columns=["seg", "chrom", "pos", "ori"])
    df["key"] = [f"{c}:{p}" for c, p in zip(df.chrom, df.pos)]  # also when no bins are left
    return df


def load_xyz(fn):
    x = np.load(fn) if fn.endswith(".npy") else np.loadtxt(fn)
    return x[:, -3:]


def load_mat(fn, b, res, norm="NONE"):
    if fn.endswith(".hic"):
        return from_hic(fn, b, res, norm)
    if fn.split("::")[0].endswith((".cool", ".mcool")):  # also file.mcool::/resolutions/5000
        return from_cool(fn, b, res, norm)
    return np.load(fn) if fn.endswith(".npy") else np.loadtxt(fn)


def ref_blocks(b):
    # per chromosome: first and last reference position used
    return {c: (g.pos.min(), g.pos.max()) for c, g in b.groupby("chrom")}


def place(b, res, get):
    # path matrix from the file; copies share contacts evenly
    n = len(b)
    m = np.zeros((n, n))
    blk = ref_blocks(b)
    for c1 in blk:
        for c2 in blk:
            s1, s2 = blk[c1][0], blk[c2][0]
            sub = get(c1, c2)
            i = b.index[b.chrom == c1].to_numpy()
            j = b.index[b.chrom == c2].to_numpy()
            r1 = ((b.pos[i] - s1) // res).to_numpy()
            r2 = ((b.pos[j] - s2) // res).to_numpy()
            m[np.ix_(i, j)] = sub[np.ix_(r1, r2)]
    k = b.key.map(b.key.value_counts()).to_numpy()
    d = np.outer(k, k).astype(float)
    same = b.key.to_numpy()[:, None] == b.key.to_numpy()[None, :]
    d[same] = (k[:, None] * (k[:, None] + 1) / 2 * np.ones(len(k)))[same]
    return m / d


def from_hic(fn, b, res, norm):
    import hicstraw
    h = hicstraw.HiCFile(fn)
    names = {c.name for c in h.getChromosomes()}
    blk = ref_blocks(b)

    def nm(c):
        return c if c in names else c.replace("chr", "")

    def rec(c1, c2):
        (a1, e1), (a2, e2) = blk[c1], blk[c2]
        n1, n2 = (e1 - a1) // res + 1, (e2 - a2) // res + 1
        sub = np.zeros((n1, n2))
        try:
            mzd = h.getMatrixZoomData(nm(c1), nm(c2), "observed", norm, "BP", res)
            recs = mzd.getRecords(a1, e1 + res - 1, a2, e2 + res - 1)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read Hi-C data for "
                f"{c1}:{a1}-{e1} and {c2}:{a2}-{e2} "
                f"at {res} bp resolution with normalization '{norm}'"
            ) from e
            
        for r in recs:
            for x, y in [(r.binX, r.binY), (r.binY, r.binX)] if c1 == c2 else [(r.binX, r.binY)]:
                x, y = (x - a1) // res, (y - a2) // res
                if 0 <= x < n1 and 0 <= y < n2:
                    sub[x, y] = r.counts
        return sub

    def get(c1, c2):
        if c1 == c2:
            return np.nan_to_num(rec(c1, c1))
        # inter-chromosomal data is stored in one order only
        return np.nan_to_num(rec(c1, c2)) + np.nan_to_num(rec(c2, c1)).T

    return place(b, res, get)


def from_cool(fn, b, res, norm):
    import cooler
    uri = fn if "::" in fn or not fn.endswith(".mcool") else f"{fn}::resolutions/{res}"
    c = cooler.Cooler(uri)
    blk = ref_blocks(b)

    def nm(x):
        return x if x in c.chromnames else x.replace("chr", "")

    def get(c1, c2):
        (a1, e1), (a2, e2) = blk[c1], blk[c2]
        bal = False if norm == "NONE" else ("weight" if norm.lower() in ("weight", "ice", "kr") else norm)
        try:
            sub = c.matrix(balance=bal).fetch((nm(c1), a1, e1 + res), (nm(c2), a2, e2 + res))
        except ValueError:
            raise SystemExit(f"{fn}: no '{bal}' balancing weights, use --norm NONE")
        return np.nan_to_num(sub)

    return place(b, res, get)


def load_contacts(fn):
    df = pd.read_csv(fn, sep=None, engine="python")
    df.columns = [c.lower() for c in df.columns]
    sc = next((c for c in ["q_value", "qvalue", "score", "p_value"] if c in df.columns), None)
    df["sc"] = df[sc] if sc else np.nan
    df["name"] = os.path.basename(fn).split(".")[0]
    return df[["bin1", "bin2", "sc", "name"]].astype({"bin1": int, "bin2": int})


def overlap(b, res, c, s, e):
    return b.index[(b.chrom == c) & (b.pos + res > s) & (b.pos < e)].tolist()


def load_feats(fn, b, res):
    # BED features -> list of (name, bins)
    df = pd.read_csv(fn, sep="\t", header=None, comment="#")
    df[0] = df[0].astype(str).map(lambda c: c if c.startswith("chr") else "chr" + c)
    df = df[df[0].isin(set(b.chrom))]
    out = []
    for r in df.itertuples(index=False):
        hit = overlap(b, res, r[0], r[1], r[2])
        if hit:
            out.append((str(r[3]) if len(r) > 3 else f"{r[0]}:{r[1]}", hit))
    return out


def gene_types(fn):
    # 6th column of the gene BED: gene type
    df = pd.read_csv(fn, sep="\t", header=None, comment="#")
    return dict(zip(df[3].astype(str), df[5].astype(str))) if df.shape[1] > 5 else {}


def load_track(fn, b, res):
    # one value per bin (BED: count, else mean)
    if re.search(r"\.(bw|bigwig)$", fn, re.I):
        import pyBigWig
        bw = pyBigWig.open(fn)
        have = bw.chroms()
        out = []
        for c, p in zip(b.chrom, b.pos):
            c = c if c in have else c.replace("chr", "")
            v = bw.stats(c, int(p), int(p + res), type="mean")[0] if c in have else None
            out.append(v or 0.0)
        return np.array(out)
    txt = "".join(l for l in open(fn) if not l.startswith(("track", "browser", "#")))
    df = pd.read_csv(io.StringIO(txt), sep="\t", header=None)
    df[[1, 2]] = df[[1, 2]].astype(int)
    df[0] = df[0].astype(str).map(lambda c: c if c.startswith("chr") else "chr" + c)
    v = np.zeros(len(b))
    graph = df.shape[1] >= 4 and pd.api.types.is_numeric_dtype(pd.to_numeric(df[3], errors="coerce").dropna())
    graph = graph and fn.lower().endswith((".bedgraph", ".bg"))
    for c, g in df.groupby(0):
        for i in b.index[b.chrom == c]:
            p = b.pos[i]
            h = g[(g[2] > p) & (g[1] < p + res)]
            if not len(h):
                continue
            if graph:
                w = np.minimum(h[2], p + res) - np.maximum(h[1], p)
                v[i] = float(np.sum(w * pd.to_numeric(h[3])) / res)
            else:
                v[i] = len(h)
    return v
