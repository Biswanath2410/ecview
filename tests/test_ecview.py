import numpy as np
import pandas as pd
import pytest

from ecview import calc, files
from ecview.main import build

RES = 5000


@pytest.fixture
def bed(tmp_path):
    # 10 bins forward, then a 4-bin copy of 20-40 kb in reverse
    f = tmp_path / "ec.bed"
    f.write_text("chrom\tstart\tend\torientation\nchr1\t0\t50000\t+\nchr1\t20000\t40000\t-\n")
    return str(f)


def test_bins_and_copies(bed):
    b = files.path_bins(files.read_bed(bed), RES)
    assert len(b) == 14
    assert list(b.pos[10:]) == [35000, 30000, 25000, 20000]
    cp = calc.copies(b)
    assert cp[4] == [13] and cp[13] == [4] and cp[0] == []


def test_collapse_keeps_total(bed):
    b = files.path_bins(files.read_bed(bed), RES)
    ref, keys = calc.ref_index(b)
    m = np.random.default_rng(0).random((14, 14))
    m = m + m.T
    c = calc.collapse(m, ref, len(keys))
    assert c.shape == (10, 10) and np.allclose(c, c.T)
    assert np.isclose(c[4, 0], m[4, 0] + m[13, 0])
    assert np.isclose(c[4, 4], m[4, 4] + m[13, 13] + m[4, 13])


def test_even_split(bed):
    b = files.path_bins(files.read_bed(bed), RES)
    m = files.place(b, RES, lambda c1, c2: np.ones((10, 10)))
    ref, keys = calc.ref_index(b)
    assert np.allclose(calc.collapse(m, ref, len(keys)), 1)


def test_ec3d_bed(tmp_path, bed):
    f = tmp_path / "ec3d.bed"
    f.write_text("#chr\tstart\tend\tcycle_id\tweight\torientation\tiscyclic\n"
                 "chr1\t0\t50000\t1\t1.0\t+\tTrue\nchr1\t20000\t40000\t1\t1.0\t-\tTrue\n")
    assert files.read_bed(str(f)).equals(files.read_bed(bed))


def test_cycles(tmp_path, bed):
    f = tmp_path / "c.txt"
    f.write_text("Segment\t1\tchr1\t0\t19999\nSegment\t2\tchr1\t20000\t39999\nSegment\t3\tchr1\t40000\t50000\n"
                 "Cycle=1;Copy_count=1;Segments=1+,2+,3+,2-\nCycle=2;Copy_count=1;Segments=0+,1+,2+,0-\n")
    c, lin = files.read_cycles(str(f), 1)
    assert not lin and len(c) == 2 and c.start[0] == 0 and c.end[0] == 50000 and c.ori[1] == "-"
    c, lin = files.read_cycles(str(f), 2)
    assert lin and len(c) == 1


def test_align_recovers_rotation():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    y = 2.5 * x @ q + 7
    z, rmsd = calc.align(y, x)
    assert rmsd < 1e-8 and np.allclose(z, x)


def test_fit_is_high_for_matching_data():
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    x = np.c_[np.cos(t), np.sin(t), 0.3 * np.sin(3 * t)]
    d = calc.dists(x) + np.eye(60)
    per, tot = calc.fit(x, d ** -2)
    assert tot > 0.99 and np.nanmin(per) > 0.9


def test_tracks(tmp_path, bed):
    b = files.path_bins(files.read_bed(bed), RES)
    g = tmp_path / "t.bedGraph"
    g.write_text('track type=bedGraph name="t"\nchr1\t0\t10000\t2\n1\t20000\t22500\t4\n')
    v = files.load_track(str(g), b, RES)
    assert v[0] == 2 and v[1] == 2 and v[4] == 2 and v[13] == 2 and v[2] == 0
    e = tmp_path / "e.bed"
    e.write_text("chr1\t100\t200\tx\nchr1\t300\t400\ty\n")
    assert files.load_track(str(e), b, RES)[0] == 2


def test_bigwig(tmp_path, bed):
    bw = pytest.importorskip("pyBigWig")
    f = str(tmp_path / "t.bw")
    h = bw.open(f, "w")
    h.addHeader([("1", 100000)])
    h.addEntries(["1"], [0], ends=[5000], values=[3.0])
    h.close()
    b = files.path_bins(files.read_bed(bed), RES)
    v = files.load_track(f, b, RES)
    assert v[0] == 3 and v[1] == 0


def test_build(tmp_path, bed):
    rng = np.random.default_rng(2)
    xyz = tmp_path / "x.txt"
    np.savetxt(xyz, rng.normal(size=(14, 3)))
    xyz2 = tmp_path / "y.txt"
    np.savetxt(xyz2, rng.normal(size=(14, 3)))
    m = rng.random((14, 14))
    np.save(tmp_path / "m.npy", m + m.T)
    pd.DataFrame({"bin1": [0, 3], "bin2": [9, 12], "q_value": [0.01, 0.02]}).to_csv(tmp_path / "si.csv", index=False)
    (tmp_path / "g.bed").write_text("chr1\t21000\t30000\tGENE1\n")
    page, log = build(bed=bed, xyz=[str(xyz), str(xyz2)], mat=str(tmp_path / "m.npy"),
                      contacts=[str(tmp_path / "si.csv")], genes=str(tmp_path / "g.bed"),
                      track=[f"g={tmp_path / 'g.bed'}"], cdn=True)
    assert "GENE1" in page and "fit to Hi-C" in page and "shift vs y" in page
    assert log[0].startswith("14 bins (ec3d binning), 10 reference bins, 8 duplicated")


def test_deconvolve_keeps_hic(tmp_path):
    # needs a CADET checkout: CADET=/path/to/cadet pytest
    import os
    from ecview import deconv
    if not os.environ.get("CADET"):
        pytest.skip("CADET not set")
    f = tmp_path / "ec.bed"
    f.write_text("chrom\tstart\tend\torientation\nchr1\t0\t500000\t+\nchr1\t200000\t300000\t-\n")
    bed = files.read_bed(str(f))
    b = files.path_bins(bed, RES)
    ref, keys = calc.ref_index(b)
    p = np.array([k[1] for k in keys]) / RES
    r = np.random.default_rng(3).poisson(2000 / (1 + np.abs(p[:, None] - p[None, :])) ** 1.2).astype(float)
    r = np.triu(r) + np.triu(r, 1).T
    m = files.place(b, RES, lambda c1, c2: r)
    np.save(tmp_path / "m.npy", m)
    assert np.allclose(calc.collapse(m, ref, len(keys)), r)
    b2, x, pi, y = deconv.run(deconv.load(None), bed, str(tmp_path / "m.npy"), RES, log=lambda s: None)
    assert np.allclose(y, r[np.ix_(*[[keys.index((c, q)) for c, q in dict.fromkeys(zip(b2.chrom, b2.pos))]] * 2)])
    r2, k2 = calc.ref_index(b2)
    assert np.allclose(calc.collapse(x, r2, len(k2)), calc.collapse(m, ref, len(keys))[np.ix_(*[[keys.index(k) for k in k2]] * 2)])


def test_binning(tmp_path):
    from ecview.main import pick_bins
    f = tmp_path / "b.bed"
    f.write_text("chr1\t1000\t20000\t+\nchr1\t31000\t39000\t-\n")
    bed = files.read_bed(str(f))
    e, c = files.path_bins(bed, RES), files.path_bins(bed, RES, "cadet")
    assert len(e) == 6 and len(c) == 5 and c.pos[0] == 0 and c.pos.iloc[-1] == 30000
    assert pick_bins(bed, RES, np.zeros((5, 3)), None, []) == "cadet"
    assert pick_bins(bed, RES, None, np.zeros((6, 6)), []) == "ec3d"
    with pytest.raises(SystemExit):
        pick_bins(bed, RES, np.zeros((7, 3)), None, [])


def test_breaks(tmp_path):
    f = tmp_path / "b.bed"
    # contiguous continuation (no break), a jump, a fold-back
    f.write_text("chr1\t0\t20000\t+\nchr1\t20000\t40000\t+\nchr1\t60000\t70000\t+\nchr1\t60000\t70000\t-\n")
    b = files.path_bins(files.read_bed(str(f)), RES)
    assert calc.breaks(b, RES) == [(7, 8), (9, 10), (11, 0)]


# structures from other methods

def path(bed):
    b = files.path_bins(files.read_bed(bed), RES)
    ref, keys = calc.ref_index(b)
    return b, ref, len(keys)


def test_struct_formats(bed, tmp_path):
    from ecview import structs
    b, ref, nr = path(bed)
    x = np.arange(42, dtype=float).reshape(14, 3) / 7
    np.savetxt(tmp_path / "a.txt", x)  # one row per path bin
    k, d = structs.read(str(tmp_path / "a.txt"))
    assert k == "rows" and np.allclose(structs.place(k, d, b, RES, ref, nr)[0], x)
    # bin x y z with a header and two bins missing
    rows = [f"{i} {x[i, 0]} {x[i, 1]} {x[i, 2]}" for i in range(14) if i not in (3, 7)]
    (tmp_path / "b.txt").write_text("bin x y z\n" + "\n".join(rows) + "\n")
    y, note = structs.place(*structs.read(str(tmp_path / "b.txt")), b, RES, ref, nr)
    assert np.isnan(y[[3, 7]]).all() and np.allclose(y[0], x[0]) and "2 bins missing" in note
    # PDB, atom order
    pdb = "".join(f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    {x[i, 0]:8.3f}{x[i, 1]:8.3f}{x[i, 2]:8.3f}\n"
                  for i in range(14))
    (tmp_path / "c.pdb").write_text(pdb)
    assert np.allclose(structs.place(*structs.read(str(tmp_path / "c.pdb")), b, RES, ref, nr)[0], x, atol=1e-3)


def test_struct_genome_order(bed, tmp_path):
    from ecview import structs
    b, ref, nr = path(bed)
    g = np.random.default_rng(1).random((nr, 3))  # one row per reference bin (no copies)
    np.savetxt(tmp_path / "g.txt", g)
    y, note = structs.place(*structs.read(str(tmp_path / "g.txt")), b, RES, ref, nr)
    assert np.allclose(y[4], y[13]) and np.allclose(y[4], g[ref[4]]) and "copies" in note
    # positions, Dip-C style
    lines = [f"1(pat)\t{p}\t{g[i, 0]}\t{g[i, 1]}\t{g[i, 2]}" for i, p in enumerate(range(0, 50000, 5000))]
    lines += [f"1(mat)\t{p}\t9\t9\t9" for p in range(0, 50000, 5000)]
    (tmp_path / "s.3dg").write_text("\n".join(lines) + "\n")
    y2 = structs.place(*structs.read(str(tmp_path / "s.3dg")), b, RES, ref, nr)[0]
    assert np.allclose(y2, y)
    # coarser beads (20 kb)
    lines = [f"chr1,{p},{p},0,0" for p in range(0, 60000, 20000)]
    (tmp_path / "c.csv").write_text("chrom,start,x,y,z\n" + "\n".join(lines) + "\n")
    y3 = structs.place(*structs.read(str(tmp_path / "c.csv")), b, RES, ref, nr)[0]
    assert y3[0, 0] == 0 and y3[5, 0] == 20000 and y3[13, 0] == 20000


def test_build_with_missing_bins(bed, tmp_path):
    x = np.cumsum(np.random.default_rng(2).normal(size=(14, 3)), 0)
    rows = [f"{i} {x[i, 0]} {x[i, 1]} {x[i, 2]}" for i in range(14) if i != 5]
    (tmp_path / "gap.txt").write_text("\n".join(rows) + "\n")
    np.savetxt(tmp_path / "full.txt", x)
    m = np.exp(-calc.dists(x))
    np.savetxt(tmp_path / "m.txt", m)
    html, log = build(bed, xyz=[str(tmp_path / "gap.txt"), f"full={tmp_path / 'full.txt'}"],
                      mat=str(tmp_path / "m.txt"), res=RES)
    t = " ".join(log)
    assert "1 bin missing" in t and "Spearman" in t and "<html" in html
    assert "full: same as --xyz" in t  # the gap-free copy aligns exactly on the shared bins


def test_several_structures(bed, tmp_path):
    import json
    import re
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(size=(14, 3)), 0)
    np.savetxt(tmp_path / "a.txt", x)
    np.savetxt(tmp_path / "b.txt", x + rng.normal(scale=0.3, size=x.shape))
    np.savetxt(tmp_path / "m.txt", np.exp(-calc.dists(x)))
    html, log = build(bed, xyz=[f"A={tmp_path / 'a.txt'}", f"B={tmp_path / 'b.txt'}"],
                      mat=str(tmp_path / "m.txt"), res=RES)
    d = json.loads(re.search(r"const D = (.*?);\nconst F3", html, re.S).group(1))
    assert [s["name"] for s in d["structs"]] == ["A", "B"]
    assert d["structs"][0]["rmsd"] == 0 and d["structs"][1]["rmsd"] > 0 and d["structs"][1]["cov"] == 14
    assert set(d["shift"]) == {"A", "B"} and len(d["shift"]["A"]["B"]) == 14
    assert np.allclose(d["shift"]["A"]["B"], d["colors"]["shift vs B"], atol=1e-4)


def test_mcool_resolution_path(bed, tmp_path):
    cooler = pytest.importorskip("cooler")
    bins = pd.DataFrame({"chrom": "chr1", "start": np.arange(0, 60000, 5000), "end": np.arange(5000, 65000, 5000)})
    i, j = np.triu_indices(12)
    pix = pd.DataFrame({"bin1_id": i, "bin2_id": j, "count": 1 + j - i})
    base = str(tmp_path / "a.cool")
    cooler.create_cooler(base, bins, pix)
    mc = str(tmp_path / "a.mcool")
    cooler.zoomify_cooler(base, mc, [5000], chunksize=100)
    b = files.path_bins(files.read_bed(bed), RES)
    a = files.load_mat(mc, b, RES)  # resolution taken from --res
    c = files.load_mat(mc + "::/resolutions/5000", b, RES)  # resolution given in the path
    assert a.shape == (14, 14) and np.allclose(a, c)


def test_short_cycle(tmp_path):
    # a 1.1 kb cycle has no 5 kb bin
    f = tmp_path / "c.txt"
    f.write_text("Segment\t1\tchr2\t15690699\t15691829\nCycle=1;Copy_count=10;Segments=1+\n")
    b, lin = files.read_cycles(str(f), 1)
    assert len(files.path_bins(b, RES)) == 0 and not lin
    m = tmp_path / "m.txt"
    np.savetxt(m, np.ones((2, 2)))
    with pytest.raises(SystemExit, match="no 5000 bp bins"):
        build(cycles=str(f), mat=str(m), res=RES)


def test_plain_arrays():
    # plotly 6 writes numpy arrays as base64; the page needs lists
    import base64
    from ecview.main import plain
    a = np.array([[1.5, np.nan], [2.0, 3.0]])
    o = {"z": {"dtype": "f8", "bdata": base64.b64encode(a.tobytes()).decode(), "shape": "2, 2"}, "x": [1, 2]}
    assert plain(o) == {"z": [[1.5, None], [2.0, 3.0]], "x": [1, 2]}


def test_path_arcs(bed):
    # arc layers: copies, and one per contact table
    from ecview import plots
    b = files.path_bins(files.read_bed(bed), RES)
    cp = calc.copies(b)
    si = pd.DataFrame({"bin1": [0, 3, 9], "bin2": [9, 12, 0]})
    f = plots.figpath(b, cp, {}, RES, {}, cons={"SI": si})
    arc = [t for t in f.data if (t.name or "").startswith("arc:")]
    assert {t.name for t in arc} == {"arc:duplicated regions", "arc:SI"}
    si = [t for t in arc if t.name == "arc:SI"]
    assert all(t.visible is False for t in si)
    dots = [t for t in si if t.mode == "markers"][0]  # one per block: 0-9 (given twice) and 3-12 are apart
    assert len(dots.x) == 2 and "2 contacts" not in " ".join(dots.hovertext)
    assert f.layout.meta["y0"] < plots.figpath(b, cp, {}, RES, {}).layout.meta["y0"]
