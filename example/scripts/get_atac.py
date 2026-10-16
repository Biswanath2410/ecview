# bigWig -> bedGraph over the path, 1 kb steps
# usage: python get_atac.py path.bed GSM7710236_D458.nodup.bw D458_atac.bedGraph
import sys

import pyBigWig

from ecview import files

bed, bw, out = sys.argv[1], pyBigWig.open(sys.argv[2]), sys.argv[3]
b = files.read_bed(bed)
have = bw.chroms()
n = 0
with open(out, "w") as f:
    f.write('track type=bedGraph name="D458 ATAC (GSM7710236)"\n')
    for c, g in b.groupby("chrom", sort=False):
        # merge the segments of this chromosome
        iv = sorted((s // 1000 * 1000, -(-e // 1000) * 1000) for s, e in zip(g.start, g.end))
        m = [list(iv[0])]
        for s, e in iv[1:]:
            if s <= m[-1][1]:
                m[-1][1] = max(m[-1][1], e)
            else:
                m.append([s, e])
        name = c if c in have else c.replace("chr", "")
        for s, e in m:
            for p in range(s, e, 1000):
                v = bw.stats(name, p, p + 1000, type="mean")[0] or 0.0
                f.write(f"{c}\t{p}\t{p + 1000}\t{v:.3f}\n")
                n += 1
print(n, "rows")
