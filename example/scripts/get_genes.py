# GENCODE genes on the path (UCSC, hg38)
# usage: python get_genes.py path.bed genes.bed [ec3d|cadet]
#        python get_genes.py cycles.txt genes.bed [ec3d|cadet] [cycle]
import json
import sys
import urllib.request

from ecview import files

bed, out = sys.argv[1], sys.argv[2]
how = sys.argv[3] if len(sys.argv) > 3 else "ec3d"
res = 5000
b = files.read_cycles(bed, int(sys.argv[4]) if len(sys.argv) > 4 else 1)[0] if bed.endswith(".txt") else files.read_bed(bed)
genes = {}
for c, s, e in zip(b.chrom, b.start, b.end):
    s, e = int(s) - 10000, int(e) + 10000
    url = f"https://api.genome.ucsc.edu/getData/track?genome=hg38;track=knownGene;chrom={c};start={s};end={e}"
    for t in json.load(urllib.request.urlopen(url))["knownGene"]:
        if t["geneType"] not in ("protein_coding", "lncRNA") or not t["geneName"] or t["geneName"].startswith("ENSG"):
            continue
        if "Ensembl_canonical" not in t.get("tag", ""):
            continue
        k = (t["chrom"], t["geneName"])
        a, z = genes.get(k, (t["chromStart"], t["chromEnd"]))[:2]  # some genes have 2 canonical rows
        genes[k] = (min(a, t["chromStart"]), max(z, t["chromEnd"]), t["strand"], t["geneType"])

bins = files.path_bins(b, res, how)
rows = sorted((c, s, e, n, st, ty) for (c, n), (s, e, st, ty) in genes.items() if files.overlap(bins, res, c, s, e))
with open(out, "w") as f:
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")
print(len(rows), "genes:", " ".join(r[3] for r in rows))
