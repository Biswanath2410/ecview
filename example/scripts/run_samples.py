# one page per ecDNA in AmpliconRepository downloads
# usage: python run_samples.py data_dir out_dir [hic.tsv] [--genes]
#   hic.tsv: sample <tab> Hi-C file; needs CADET=<folder>
import glob
import os
import re
import subprocess
import sys
import zipfile

import pandas as pd

from ecview import files
from ecview.main import build

args = [a for a in sys.argv[1:] if not a.startswith("--")]
data, out = args[:2]
hic = {}
if len(args) > 2:
    hic = dict(l.rstrip("\n").split("\t")[:2] for l in open(args[2]) if l.strip() and not l.startswith("#"))
os.makedirs(out, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))


def cycles(txt):
    # -> {cycle: (copy count, circular)}
    cy = {}
    for l in txt.splitlines():
        if l.startswith("Cycle="):
            d = dict(x.split("=", 1) for x in l.strip().split(";"))
            cy[int(d["Cycle"])] = (float(d["Copy_count"]), not any(s.startswith("0") for s in d["Segments"].split(",")))
    return cy


def cover(b, reg):
    hit = 0
    for r in reg.itertuples():
        x = b[b.chrom == r.c]
        hit += sum(max(0, min(e, r.e) - max(s, r.s)) for s, e in zip(x.start, x.end))
    return hit / (reg.e - reg.s).sum()


rows = []
for z in sorted(glob.glob(os.path.join(data, "*.zip"))):
    zf = zipfile.ZipFile(z)
    names = zf.namelist()
    for bed in sorted(n for n in names if re.search(r"_amplicon\d+_ecDNA_\d+\.bed$", n)):
        s, amp = re.match(r"(.+)_amplicon(\d+)_ecDNA", os.path.basename(bed)).groups()
        reg = pd.read_csv(zf.open(bed), sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"])
        cn = next(n for n in names if os.path.basename(n) == f"{s}_amplicon{amp}_cycles.txt")
        txt = zf.read(cn).decode()
        fn = os.path.join(out, os.path.basename(cn))
        open(fn, "w").write(txt)
        # best coverage, then highest copy count
        cand = []
        for k, (cc, circ) in cycles(txt).items():
            if circ:
                cand.append((cover(files.read_cycles(fn, k)[0], reg), cc, k))
        best = max(c[0] for c in cand)
        cov, cc, k = max((c for c in cand if c[0] >= best - 0.05), key=lambda c: c[1])
        b = files.read_cycles(fn, k)[0]
        name = f"{s}_amplicon{amp}"
        row = dict(sample=s, amplicon=amp, cycle=k, copies=round(cc, 1), segments=len(b),
                   Mb=round(float((b.end - b.start).sum()) / 1e6, 2), covers=f"{cov:.0%}", page="")
        if s in hic:
            g = None
            if "--genes" in sys.argv:
                g = os.path.join(out, name + "_genes.bed")
                subprocess.run([sys.executable, os.path.join(here, "get_genes.py"), fn, g, "cadet", str(k)], check=True)
            html, log = build(cycles=fn, cycle=k, mat=hic[s], deconvolve=True, genes=g, title=name)
            open(os.path.join(out, name + ".html"), "w").write(html)
            row["page"] = name + ".html"
            row["bins"] = log[0].split()[0]
        rows.append(row)
        print(row, flush=True)

t = pd.DataFrame(rows)
t.to_csv(os.path.join(out, "samples.tsv"), sep="\t", index=False)
print(t.to_string(index=False))
