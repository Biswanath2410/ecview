# ENCODE enhancer-like cCREs on the path (UCSC, hg38)
# usage: python get_enhancers.py path.bed enhancers.bed
import json
import sys
import urllib.request

from ecview import files

bed, out = sys.argv[1], sys.argv[2]
b = files.read_bed(bed)
rows = set()
for c, g in b.groupby("chrom"):
    s, e = int(g.start.min()) - 10000, int(g.end.max()) + 10000
    url = f"https://api.genome.ucsc.edu/getData/track?genome=hg38;track=encodeCcreCombined;chrom={c};start={s};end={e}"
    for t in json.load(urllib.request.urlopen(url))["encodeCcreCombined"]:
        lab = t.get("ucscLabel") or t.get("encodeLabel")
        if lab in ("enhP", "enhD"):
            rows.add((t["chrom"], t["chromStart"], t["chromEnd"], t["name"]))
with open(out, "w") as f:
    for r in sorted(rows):
        f.write("\t".join(map(str, r)) + "\n")
print(len(rows), "enhancer-like elements")
