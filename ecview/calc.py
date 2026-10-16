# copies, genome-space map, fit to Hi-C, structure alignment
import numpy as np
import pandas as pd


def copies(b):
    grp = b.groupby("key").groups
    return [[int(j) for j in grp[k] if j != i] for i, k in enumerate(b.key)]


def breaks(b, res, linear=False):
    # (i, i + 1) at junctions, as in ec3D
    n, c, p = len(b), b.chrom.to_numpy(), b.pos.to_numpy()
    out = []
    for i in range(n if not linear else n - 1):
        j = (i + 1) % n
        if c[i] != c[j] or abs(p[j] - p[i]) != res:
            out.append((i, j))
        elif 0 < i < n - 2 and c[i - 1] == c[i] and c[j] == c[j + 1]:
            a, e = p[i] - p[i - 1], p[j + 1] - p[j]
            if abs(a) == res and abs(e) == res and a != e:
                out.append((i, j))
    return out


def ref_index(b):
    # ref[i] = reference bin of path bin i
    def chrnum(c):
        c = c.replace("chr", "")
        return (0, int(c)) if c.isdigit() else (1, c)
    keys = sorted(set(zip(b.chrom, b.pos)), key=lambda x: (chrnum(x[0]), x[1]))
    k2r = {k: r for r, k in enumerate(keys)}
    ref = np.array([k2r[k] for k in zip(b.chrom, b.pos)])
    return ref, keys


def collapse(m, ref, nr):
    # copies summed into genome order
    i, j = np.triu_indices(len(m))
    c = np.zeros((nr, nr))
    np.add.at(c, (ref[i], ref[j]), m[i, j])
    return c + c.T - np.diag(np.diag(c))


def spacing(xyz):
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return float(np.nanmedian(d))  # bins missing from a structure are NaN


def dists(xyz):
    return np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)


def spear(a, b):
    a, b = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


def fit(xyz, m, linear=False, gap=2):
    # per bin: Spearman of -3D distance with contacts
    n = len(xyz)
    d = dists(xyz)
    k = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    if not linear:
        k = np.minimum(k, n - k)
    have = ~np.isnan(xyz[:, 0])
    out = np.full(n, np.nan)
    for i in np.where(have)[0]:
        ok = (k[i] >= gap) & have
        if ok.sum() > 10 and m[i, ok].std() > 0:
            out[i] = spear(-d[i, ok], m[i, ok])
    ok = (k >= gap) & have[:, None] & have[None, :]
    return out, spear(-d[ok], m[ok])


def align(x, ref):
    # superimpose x on ref (mirror allowed), on shared bins
    ok = ~np.isnan(x[:, 0]) & ~np.isnan(ref[:, 0])
    mx, mr = x[ok].mean(0), ref[ok].mean(0)
    a, b = ref[ok] - mr, x[ok] - mx
    u, s, vt = np.linalg.svd(b.T @ a)
    r = u @ vt
    sc = s.sum() / (b ** 2).sum()
    y = sc * (x - mx) @ r + mr
    return y, float(np.sqrt(((y[ok] - ref[ok]) ** 2).sum(1).mean()))


def spread(xs):
    # per bin distance from the mean structure
    x = np.stack(xs)
    return np.sqrt(np.nanmean(((x - np.nanmean(x, 0)) ** 2).sum(2), 0))
