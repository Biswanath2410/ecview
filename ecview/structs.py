# 3D structures from any method, placed on the path bins
import numpy as np
import pandas as pd


def num1(v):
    # same as np.loadtxt, to the last digit
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def read(fn):
    # -> (kind, data): rows, bins or pos
    if fn.endswith(".npy"):
        return "rows", np.load(fn)[:, -3:].astype(float)
    if fn.lower().endswith(".pdb"):
        x = [[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in open(fn)
             if l.startswith(("ATOM", "HETATM"))]
        return "rows", np.array(x)
    sep = "," if fn.endswith(".csv") else r"\s+"
    df = pd.read_csv(fn, sep=sep, header=None, comment="#", engine="python", dtype=str)
    num = df.map(num1)
    if num.iloc[0, -3:].isna().any():
        df, num = df.iloc[1:].reset_index(drop=True), num.iloc[1:].reset_index(drop=True)  # header line
    if num[0].isna().any():  # first column is a chromosome
        c = df[0]
        hap = c.str.extract(r"\((\w+)\)$")[0]
        if hap.notna().any():  # Dip-C: two parental copies; keep the first one
            keep = (hap == hap.dropna().iloc[0]).to_numpy()
            c, num = c[keep], num[keep]
        c = c.str.replace(r"\(.*\)$", "", regex=True)
        c = c.map(lambda s: s if s.startswith("chr") else "chr" + s)
        return "pos", (c.to_numpy(), num[1].to_numpy(float), num.iloc[:, -3:].to_numpy(float))
    a = num.to_numpy(float)
    if a.shape[1] == 3:
        return "rows", a
    if a.shape[1] == 4 and np.all(a[:, 0] == np.round(a[:, 0])):
        return "bins", (a[:, 0].astype(int), a[:, 1:])
    return "rows", a[:, -3:]


def place(kind, data, b, res, ref, nref):
    # -> (xyz per path bin, NaN if missing; note)
    n = len(b)
    out = np.full((n, 3), np.nan)
    if kind == "rows":
        x = data
        if len(x) == n:
            return x, ""
        if len(x) == nref:
            return x[ref], "genome order: copies share a position"
        raise SystemExit(f"{len(x)} coordinates, but {n} path bins and {nref} reference bins")
    if kind == "bins":
        idx, x = data
        if idx.min() < 0 or idx.max() >= n:
            raise SystemExit(f"bin numbers {idx.min()}-{idx.max()} outside the {n} path bins")
        out[idx] = x
        miss = n - len(set(idx))
        return out, f"{miss} bin{'s' * (miss != 1)} missing" if miss else ""
    c, p, x = data
    for ch in set(b.chrom):
        sel = c == ch
        if not sel.any():
            continue
        cp, cx = p[sel], x[sel]
        o = np.argsort(cp)
        cp, cx = cp[o], cx[o]
        step = np.median(np.diff(cp)) if len(cp) > 1 else res
        for i in np.where(b.chrom.to_numpy() == ch)[0]:
            s = b.pos.iat[i]
            if step <= res:  # finer than the bins: mean of the beads inside
                k = (cp >= s) & (cp < s + res)
                if k.any():
                    out[i] = cx[k].mean(0)
            else:  # coarser: the bead whose interval holds the bin centre
                j = np.searchsorted(cp, s + res / 2, side="right") - 1
                if j >= 0 and s + res / 2 < cp[j] + step:
                    out[i] = cx[j]
    miss = int(np.isnan(out[:, 0]).sum())
    note = "genome positions: copies share a position"
    return out, note + (f", {miss} bin{'s' * (miss != 1)} missing" if miss else "")
