"""Check the graph measurements on generated images whose true network is known.

    python validate.py        # ~3 minutes; writes results/validation.json and images/validation.png

30 unseen images: 3 surfaces (mud, glaze, paint) x 2 ways of forming (sequential = T
junctions, simultaneous = Y junctions), 5 random seeds each. Default settings throughout.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from network import Params, analyse
from synth import make_example

CONFIGS = [("mud", "sequential", 120), ("mud", "voronoi", 110), ("glaze", "sequential", 75),
           ("glaze", "voronoi", 80), ("paint", "sequential", 70), ("paint", "voronoi", 90)]
MARGIN, MATCH_PX = 12, 10


def match_junctions(truth, net, H, W):
    """Pair true and measured junctions within MATCH_PX (an X node may stand for two close truths)."""
    def inside(x, y):
        return MARGIN <= x < W - MARGIN and MARGIN <= y < H - MARGIN
    tj = [(x, y, t) for x, y, t in truth["junction_xy_type"] if inside(x, y)]
    mj = [(j["x"], j["y"], j["type"]) for j in net.junctions if inside(j["x"], j["y"])]
    if not tj or not mj:
        return dict(n_true=len(tj), n_meas=len(mj), recall=0.0, precision=0.0, type_pairs=[])
    tree = cKDTree([(x, y) for x, y, _ in mj])
    used = {}
    pairs = []
    for x, y, t in tj:
        d, i = tree.query((x, y))
        cap = 2 if mj[i][2] == "X" else 1
        if d <= MATCH_PX and used.get(i, 0) < cap:
            used[i] = used.get(i, 0) + 1
            pairs.append((t, mj[i][2]))
    return dict(n_true=len(tj), n_meas=len(mj), recall=len(pairs) / len(tj), precision=len(used) / len(mj),
                type_pairs=pairs)


def main(seeds=range(100, 105)):
    rows = []
    for surface, network, cell in CONFIGS:
        for seed in seeds:
            img, truth = make_example(surface, network, cell, seed)
            H, W = img.shape[:2]
            net = analyse(img, Params())
            s = net.summary()
            m = match_junctions(truth, net, H, W)
            ty = [(t, p) for t, p in m["type_pairs"] if p in ("T", "Y")]
            rows.append(dict(surface=surface, network=network, seed=seed,
                             length_err_pct=100 * (s["total_crack_length_px"] / truth["total_length_px"] - 1),
                             true_cells=truth["closed_cells"], cells=s["closed_cells"],
                             junction_recall=m["recall"], junction_precision=m["precision"],
                             type_correct=sum(t == p for t, p in ty), type_total=len(ty),
                             true_T_fraction=truth["T_junctions"] / max(truth["junctions"], 1),
                             T_fraction=s["T_fraction"], corners=s["mean_cell_corners"],
                             polarity_ok=(s["polarity"] == ("light" if surface == "paint" else "dark"))))
            r = rows[-1]
            print(f"{surface:5s} {network:10s} {seed}: length {r['length_err_pct']:+5.1f} %  junctions "
                  f"R {r['junction_recall']:.2f} P {r['junction_precision']:.2f}  cells {r['cells']}/{r['true_cells']}"
                  f"  T-frac {r['T_fraction']:.2f} (true {r['true_T_fraction']:.2f})  corners {r['corners']}")

    def agg(sel):
        sub = [r for r in rows if sel(r)]
        cells_err = [abs(r["cells"] - r["true_cells"]) / max(r["true_cells"], 1) for r in sub]
        return {"images": len(sub),
                "total_length_error_%_median_abs": round(float(np.median([abs(r["length_err_pct"]) for r in sub])), 1),
                "total_length_error_%_mean": round(float(np.mean([r["length_err_pct"] for r in sub])), 1),
                "junction_recall": round(float(np.mean([r["junction_recall"] for r in sub])), 3),
                "junction_precision": round(float(np.mean([r["junction_precision"] for r in sub])), 3),
                "junction_type_accuracy": round(sum(r["type_correct"] for r in sub) /
                                                max(sum(r["type_total"] for r in sub), 1), 3),
                "closed_cells_error_%_median": round(100 * float(np.median(cells_err)), 1),
                "T_fraction_mean": round(float(np.mean([r["T_fraction"] for r in sub])), 2),
                "mean_cell_corners": round(float(np.mean([r["corners"] for r in sub])), 2),
                "polarity_correct": f"{sum(r['polarity_ok'] for r in sub)}/{len(sub)}"}

    out = {"all": agg(lambda r: True),
           "sequential (T)": agg(lambda r: r["network"] == "sequential"),
           "voronoi (Y)": agg(lambda r: r["network"] == "voronoi")}
    for s in ("mud", "glaze", "paint"):
        out[s] = agg(lambda r, s=s: r["surface"] == s)
    Path("results").mkdir(exist_ok=True)
    Path("results/validation.json").write_text(json.dumps({"summary": out, "images": rows}, indent=1))
    print(json.dumps(out, indent=2))

    # figure
    Path("images").mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = {"mud": "#8d6748", "glaze": "#3d9a91", "paint": "#c0392b"}
    for k, (net_type, lab) in enumerate([("sequential", "formed one after another\n(T junctions)"),
                                         ("voronoi", "formed at the same time\n(Y junctions)")]):
        for surf in cols:
            v = [r["T_fraction"] for r in rows if r["network"] == net_type and r["surface"] == surf]
            ax[0].scatter(np.full(len(v), k) + np.linspace(-0.12, 0.12, len(v)) + {"mud": -0.25, "glaze": 0,
                          "paint": 0.25}[surf], v, color=cols[surf], s=30, label=surf if k == 0 else None)
            c = [r["corners"] for r in rows if r["network"] == net_type and r["surface"] == surf]
            ax[1].scatter(np.full(len(c), k) + np.linspace(-0.12, 0.12, len(c)) + {"mud": -0.25, "glaze": 0,
                          "paint": 0.25}[surf], c, color=cols[surf], s=30)
    for a, t, yl in [(ax[0], "Measured share of T junctions", (0, 1)), (ax[1], "Measured corners per cell", (3, 6.5))]:
        a.set_xticks([0, 1], ["formed one after another\n(true: 100 % T)", "formed at the same time\n(true: 0 % T)"])
        a.set_title(t)
        a.set_ylim(*yl)
        a.set_xlim(-0.6, 1.6)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].axhline(0.5, color="#999", lw=0.8, ls="--")
    ax[0].legend(frameon=False, fontsize=9)
    tl = [r["length_err_pct"] for r in rows]
    ax[2].scatter([r["true_cells"] for r in rows], [r["cells"] for r in rows],
                  c=[cols[r["surface"]] for r in rows], s=30)
    mx = max(max(r["true_cells"] for r in rows), max(r["cells"] for r in rows)) * 1.05
    ax[2].plot([0, mx], [0, mx], "k--", lw=1)
    ax[2].set(xlabel="true number of closed cells", ylabel="measured", xlim=(0, mx), ylim=(0, mx),
              title=f"Closed cells  ·  total length error {np.median(np.abs(tl)):.1f} % (median)")
    ax[2].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("images/validation.png", dpi=110)


if __name__ == "__main__":
    main()
