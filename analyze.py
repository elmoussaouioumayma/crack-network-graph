"""Analyse crack network images from the command line.

    python analyze.py examples/dried_mud.png
    python analyze.py photos/*.jpg --px-per-mm 12 --out results/batch

For each image, writes <name>_graph.png, <name>_cells.png, <name>_charts.png,
<name>.geojson, <name>_segments.csv, <name>.graphml and <name>_summary.json, and a
summary.csv with one row per image.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2

from network import Params, analyse
from viz import cells_image, charts, edges_csv, overlay, to_geojson, to_graphml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--polarity", choices=["auto", "dark", "light"], default="auto")
    ap.add_argument("--max-width", type=int, default=Params.max_width_px)
    ap.add_argument("--sensitivity", type=float, default=Params.sensitivity)
    ap.add_argument("--prune", type=float, default=Params.prune_px)
    ap.add_argument("--bridge", type=float, default=Params.bridge_px)
    ap.add_argument("--px-per-mm", type=float, default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    p = Params(a.polarity, a.max_width, a.sensitivity, Params.min_area_px, a.prune, a.bridge, a.px_per_mm)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in map(Path, a.images):
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        net = analyse(rgb, p)
        s = net.summary()
        n = path.stem
        cv2.imwrite(str(out / f"{n}_graph.png"), cv2.cvtColor(overlay(rgb, net), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{n}_cells.png"), cv2.cvtColor(cells_image(net), cv2.COLOR_RGB2BGR))
        charts(net).savefig(out / f"{n}_charts.png", dpi=100)
        (out / f"{n}.geojson").write_text(to_geojson(net))
        (out / f"{n}_segments.csv").write_text(edges_csv(net))
        (out / f"{n}_summary.json").write_text(json.dumps(s, indent=2))
        to_graphml(net, out / f"{n}.graphml")
        rows.append({"image": path.name, **s})
        print(f"{path.name}: {s['segments']} segments, {s['junctions']} junctions "
              f"(T {s['T_junctions']}, Y {s['Y_junctions']}, X {s['X_junctions']}), {s['closed_cells']} cells")
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
