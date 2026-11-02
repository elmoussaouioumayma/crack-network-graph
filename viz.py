"""Pictures and exports of a measured crack network."""

import csv
import io
import json

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

JUNCTION_COLOURS = {"T": (255, 140, 0), "Y": (30, 144, 255), "X": (186, 85, 211)}
CORNER_COLOURS = {3: "#f4a259", 4: "#e9c46a", 5: "#8ab17d", 6: "#2a9d8f", 7: "#264653"}


def _orientation_colour(deg):
    """Cyclic colour for a direction 0..180 deg (0 and 180 are the same direction)."""
    import colorsys
    return tuple(int(255 * c) for c in colorsys.hsv_to_rgb((deg % 180) / 180.0, 0.75, 1.0))


def overlay(rgb, net, line=2, by_direction=False):
    """Image with the vector network (edges light, or coloured by direction) and the
    junctions coloured by type."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    out = np.repeat((0.35 * grey + 20)[..., None], 3, 2).astype(np.uint8).copy()   # dark, colourless background
    G = net.graph
    for u, v, d in G.edges(data=True):
        e = d["path"][:, ::-1].astype(np.int32)                      # (x, y) for OpenCV
        dy, dx = d["path"][-1] - d["path"][0]
        col = _orientation_colour(float(np.rad2deg(np.arctan2(-dy, dx)) % 180)) if by_direction else (235, 235, 225)
        cv2.polylines(out, [e], False, col, line, cv2.LINE_AA)
    r = max(5, line + 4)
    for n, nd in G.nodes(data=True):
        y, x = (int(round(c)) for c in nd["pos"])
        t = nd.get("type")
        if t in JUNCTION_COLOURS:
            cv2.circle(out, (x, y), r, JUNCTION_COLOURS[t], -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), r, (255, 255, 255), 1, cv2.LINE_AA)
        elif nd["kind"] == "end":
            cv2.circle(out, (x, y), r - 1, (230, 230, 230), 1, cv2.LINE_AA)
    return out


def cells_image(net):
    """Closed cells coloured by their number of corners (3 ... 7+); open regions grey."""
    H, W = net.cells.shape
    out = np.full((H, W, 3), 235, np.uint8)
    lut = {k: tuple(int(CORNER_COLOURS[k][i:i + 2], 16) for i in (1, 3, 5)) for k in CORNER_COLOURS}
    for c in net.cell_stats:
        k = int(np.clip(c["corners"], 3, 7))
        out[net.cells == c["id"]] = lut[k]
    out[cv2.dilate(net.skeleton.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (40, 40, 40)
    return out


def charts(net):
    """Orientation rose, junction angles, cell corners and cell sizes in one figure."""
    fig = plt.figure(figsize=(16, 4.4))
    u = net.unit

    ax = fig.add_subplot(1, 4, 1, projection="polar")
    th = np.deg2rad([e["orientation_deg"] for e in net.edges])
    w = np.array([net.L(e["length"]) for e in net.edges])
    bins = np.linspace(0, np.pi, 19)
    h, _ = np.histogram(th, bins, weights=w)
    centers = (bins[:-1] + bins[1:]) / 2
    width = bins[1] - bins[0]
    for c, val in zip(np.r_[centers, centers + np.pi], np.r_[h, h]):
        ax.bar(c, val, width=width, color=np.array(_orientation_colour(np.rad2deg(c) % 180)) / 255,
               edgecolor="white", linewidth=0.5)
    ax.set_yticklabels([])
    ax.set_title(f"Crack directions (by length)\nanisotropy {net.summary()['orientation_anisotropy']}", fontsize=10, pad=14)

    ax = fig.add_subplot(1, 4, 2)
    angles = [a for j in net.junctions if j["degree"] == 3 for a in j["angles_deg"]]
    ax.hist(angles, bins=np.arange(0, 241, 10), color="#7a8b99", edgecolor="white")
    for x, lab in [(90, "T"), (120, "Y"), (180, "T")]:
        ax.axvline(x, color=np.array(JUNCTION_COLOURS[lab]) / 255, lw=2, ls="--")
    s = net.summary()
    ax.set_title(f"Angles at 3-way junctions\nT {s['T_junctions']}  ·  Y {s['Y_junctions']}  ·  X {s['X_junctions']}",
                 fontsize=10)
    ax.set_xlabel("angle between neighbouring cracks (°)")

    ax = fig.add_subplot(1, 4, 3)
    k = np.clip([c["corners"] for c in net.cell_stats], 3, 7)
    counts = [int((k == i).sum()) for i in range(3, 8)]
    ax.bar(["3", "4", "5", "6", "7+"], counts, color=[CORNER_COLOURS[i] for i in range(3, 8)])
    ax.set_title(f"Corners per closed cell\nmean {s['mean_cell_corners']}", fontsize=10)
    ax.set_xlabel("corners")

    ax = fig.add_subplot(1, 4, 4)
    areas = [net.A(c["area"]) for c in net.cell_stats]
    if areas:
        ax.hist(areas, bins=15, color="#2a9d8f", edgecolor="white")
    ax.set_title(f"Cell size\nmedian {s.get(f'median_cell_area_{u}2', 0)} {u}²", fontsize=10)
    ax.set_xlabel(f"area ({u}²)")
    for a in fig.axes[1:]:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)                      # room for the title of the polar plot
    return fig


# ----------------------------------------------------------------------------- exports
def _xy(net, path):
    s = 1 / net.px_per_mm if net.px_per_mm else 1.0
    return [[round(float(x) * s, 3), round(float(y) * s, 3)] for y, x in path]


def to_geojson(net):
    """Edges as LineStrings and junctions as Points (image coordinates: x right, y down)."""
    feats = []
    for e, (_, _, d) in zip(net.edges, net.graph.edges(data=True)):
        props = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items() if k not in ("u", "v")}
        props.update(from_node=e["u"], to_node=e["v"], unit=net.unit,
                     length=round(net.L(e["length"]), 3), width=round(net.L(e["width"]), 3))
        feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": _xy(net, d["path"])},
                      "properties": props})
    for j in net.junctions:
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point", "coordinates": _xy(net, [(j["y"], j["x"])])[0]},
                      "properties": {"node": j["node"], "junction_type": j["type"], "degree": j["degree"],
                                     "angles_deg": j["angles_deg"]}})
    return json.dumps({"type": "FeatureCollection", "features": feats})


def edges_csv(net):
    buf = io.StringIO()
    w = csv.writer(buf)
    u = net.unit
    w.writerow(["segment", "from_node", "to_node", f"length_{u}", f"width_{u}", "orientation_deg", "tortuosity"])
    for e in net.edges:
        w.writerow([e["id"], e["u"], e["v"], round(net.L(e["length"]), 3), round(net.L(e["width"]), 3),
                    round(e["orientation_deg"], 1), round(e["tortuosity"], 3)])
    return buf.getvalue()


def to_graphml(net, path):
    G = nx.Graph()
    for n, d in net.graph.nodes(data=True):
        G.add_node(n, y=float(d["pos"][0]), x=float(d["pos"][1]), kind=d["kind"], type=d.get("type", ""))
    for e in net.edges:
        G.add_edge(e["u"], e["v"], length=float(net.L(e["length"])), width=float(net.L(e["width"])),
                   orientation_deg=float(e["orientation_deg"]))
    nx.write_graphml(G, path)
