# ec3D v1.0.0, keeping the random starts in path order
# usage: python run_ec3d.py sample.hic sample_ecDNA.bed out/prefix [resolution]
import os
import sys

import hic2cool
import numpy as np
from threadpoolctl import threadpool_limits
from ec3d.extract_matrix import extract_matrix
from ec3d.spatial_structure import reconstruct_3D_structure
from ec3d.expand_matrix import expand_matrix
from ec3d.significant_interactions import identify_significant_interactions


def main():
    hic, bed, p = sys.argv[1], sys.argv[2], sys.argv[3]
    res = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    if hic.endswith(".hic"):
        cool = p + ".cool"
        hic2cool.hic2cool_convert(hic, cool, res)
        hic = cool
    extract_matrix(hic, bed, res, p, p + "_preprocessing.log")
    with threadpool_limits(limits=8, user_api="blas"):
        reconstruct_3D_structure(p + "_collapsed_matrix.txt", p + "_annotations.bed", p, save_repeats=True)
    prm = dict(l.strip().split("\t") for l in open(p + "_hyperparameters.txt"))
    expand_matrix(p + "_raw_collapsed_matrix.txt", p + "_annotations.bed", p + "_coordinates.txt",
                  float(prm["alpha"]), float(prm["beta"]), p)
    identify_significant_interactions(p, p + "_expanded_matrix.txt")

    # match rows to put the repeats in path order
    x = np.loadtxt(p + "_coordinates.txt")
    reps = [np.loadtxt(f"{p}_repeat{k}_coordinates.txt") for k in range(1, 6)]
    best = min(reps, key=lambda r: np.abs(np.sort(r, 0) - np.sort(x, 0)).max())
    idx = ((x[:, None] - best[None]) ** 2).sum(2).argmin(1)
    for k, r in enumerate(reps, 1):
        np.savetxt(f"{p}_repeat{k}_ordered_coordinates.txt", r[idx])
    print("done; *_ordered_coordinates.txt are in path order (run2 in the example = repeat1)")


if __name__ == "__main__":  # hic2cool starts worker processes (macOS re-runs the script)
    main()
