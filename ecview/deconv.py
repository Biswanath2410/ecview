# copy-aware deconvolution with CADET Layer 1 (https://github.com/Biswanath2410/cadet)
import importlib.util
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

from . import calc, files


def load(cadet):
    # cadet: CADET folder, its scripts/ folder, or cadet_layer1.py
    p = os.path.abspath(os.path.expanduser(cadet or os.environ.get("CADET", "")))
    for f in [p, os.path.join(p, "cadet_layer1.py"), os.path.join(p, "scripts", "cadet_layer1.py")]:
        if f.endswith(".py") and os.path.exists(f):
            break
    else:
        raise SystemExit("--deconvolve needs CADET: git clone https://github.com/Biswanath2410/cadet "
                         "and give --cadet <folder> (or set CADET)")
    sys.path.insert(0, os.path.dirname(f))
    spec = importlib.util.spec_from_file_location("cadet_layer1", f)
    cd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cd)
    return cd


def run(cd, bed, mat, res, norm="NONE", log=print):
    pdf, _ = cd.build_path_bin_table(bed[["chrom", "start", "end"]].values.tolist(), bed.ori.tolist(), res)
    pdf, rdf, members = cd.build_refbin_index(pdf)
    b = pd.DataFrame({"seg": pdf.seg_idx, "chrom": pdf.chrom, "pos": pdf.genomic_bp, "ori": pdf.strand})
    b["key"] = b.chrom + ":" + b.pos.astype(str)
    n = len(b)

    # observed reference matrix Y, in CADET's reference order
    m = files.load_mat(mat, b, res, norm)
    y = calc.collapse(m, pdf.ref_id.to_numpy(), len(rdf))

    _, bad = cd.detect_bad_bins(cd.sym(y), cd.DEFAULT_MAD_THRESH)
    fam = cd.build_families(rdf, members)
    d0, c0 = cd.initial_fit_pairs(pdf, rdf, y, res, bad_ref_set=set(bad))
    if len(d0) < 10:
        raise SystemExit(f"only {len(d0)} clean pairs for the backbone fit")
    tmp = tempfile.mkdtemp(prefix="ecview_")
    decay = cd.fit_initial_decay(d0, c0, out_prefix=os.path.join(tmp, "init"))
    zp, _ = cd.calibrate_zero_proxy(y, rdf)
    bb = cd.build_backbone_matrix(n, res, decay, zp)
    same = cd.build_same_ref_mask(pdf)
    a0 = cd.init_alpha(n)
    dup = bool((rdf.multiplicity > 1).any())
    pi = cd.compute_copy_anchors(y, pdf, bb, window_bins=50, eps=1e-3) if dup else None
    log(f"CADET Layer 1: {n} bins, {len(rdf)} reference bins, EM ...")
    lam, alpha, x, _, hist = cd.run_em(y, fam, n, bb, cd.build_lambda(bb, a0), a0, tmp, max_iter=100, tol=1e-6,
                                       tau=1.0, alpha_min=0.1, alpha_max=10.0, same_ref_mask=same,
                                       tau_between_copy=10.0, pi=pi, verbose=False)
    shutil.rmtree(tmp, ignore_errors=True)
    err = np.abs(cd.collapse_to_ref(x, pdf, rdf) - y).max()
    log(f"CADET Layer 1: {len(hist)} EM steps, recollapse error {err:.2g}")
    return b, cd.sym(x), pi, y
