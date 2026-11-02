"""Turn an image of a crack network into a graph and measure it.

    segment()      image -> binary crack mask (black-hat / top-hat filter + hysteresis)
    skeleton_graph mask -> 1-pixel centre lines -> graph (nodes = junctions and crack ends,
                   edges = crack segments with their pixel path), with spur pruning and
                   bridging of small gaps
    measure()      graph -> lengths, widths, orientations, junction angles (T / Y / X),
                   closed cells (pieces between cracks) and network-level statistics

The graph is a networkx.MultiGraph; every edge stores its ordered pixel path, so the
network can be exported as vectors (GeoJSON, GraphML) instead of pixels.
"""

from dataclasses import dataclass, field

import cv2
import networkx as nx
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter1d
from skimage import measure
from skimage.filters import apply_hysteresis_threshold, threshold_otsu
from skimage.morphology import remove_small_objects, skeletonize

NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class Params:
    polarity: str = "auto"        # "dark" (cracks darker than the surface), "light", or "auto"
    max_width_px: int = 15        # widest crack; sets the size of the black-hat filter
    sensitivity: float = 1.0      # > 1 finds fainter cracks, < 1 only clear ones
    min_area_px: int = 60         # smaller detected pieces are noise
    prune_px: float = 12.0        # dead-end branches shorter than this are removed (skeleton spurs)
    bridge_px: float = 15.0       # a crack end is joined to a crack ahead of it if the gap is shorter
    px_per_mm: float = None       # scale; without it everything is in pixels


# ----------------------------------------------------------------------------- segmentation
def _response(grey, size, dark):
    """Black-hat (dark cracks) or top-hat (light cracks): structures narrower than `size`."""
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    op = cv2.MORPH_BLACKHAT if dark else cv2.MORPH_TOPHAT
    return cv2.morphologyEx(cv2.GaussianBlur(grey, (0, 0), 1.2), op, se).astype(np.float32)


def _threshold(resp, sensitivity, min_area):
    """Hysteresis: Otsu level starts a crack, 65 % of it continues one (keeps faint parts
    that are connected to clear cracks, but not isolated texture)."""
    hi = threshold_otsu(resp) / sensitivity
    mask = apply_hysteresis_threshold(resp, 0.65 * hi, hi)
    mask = cv2.medianBlur(mask.astype(np.uint8), 3).astype(bool)    # smooth ragged edges
    return remove_small_objects(mask, max_size=min_area)


def _tail_score(resp):
    """How far the strongest responses stand out from the typical one (in robust std units).
    Cracks are rare and strong, so the correct polarity has the heavier tail."""
    med = np.median(resp)
    mad = np.median(np.abs(resp - med)) + 1e-6
    return (np.percentile(resp, 99) - med) / mad


def to_grey(rgb):
    """Colour image -> one channel along the colour direction with the most variation
    (first principal component). Cracks often differ from the surface in colour more than
    in brightness, e.g. pale wood showing through red paint."""
    if rgb.ndim == 2:
        return rgb.astype(np.float32)
    x = rgb.reshape(-1, 3).astype(np.float32)
    x = x - x.mean(0)
    w = np.linalg.eigh(np.cov(x[::7].T))[1][:, -1]
    w = w if w.sum() >= 0 else -w                                # + = brighter
    g = x @ w
    lo, hi = np.percentile(g, [0.5, 99.5])
    return np.clip((g - lo) / max(hi - lo, 1e-6) * 255, 0, 255).reshape(rgb.shape[:2]).astype(np.float32)


def segment(rgb, p: Params):
    grey = to_grey(rgb)
    size = 2 * p.max_width_px + 1
    if p.polarity == "auto":
        dark = _tail_score(_response(grey, size, True)) >= _tail_score(_response(grey, size, False))
    else:
        dark = p.polarity == "dark"
    mask = _threshold(_response(grey, size, dark), p.sensitivity, p.min_area_px)
    # drop round spots (bubbles, pores, pebbles); keep line-like pieces and connected networks
    lab = measure.label(mask)
    keep = np.zeros_like(mask)
    for r in measure.regionprops(lab):
        elong = r.axis_major_length / max(r.axis_minor_length, 1)
        if elong >= 3 or r.area > 30 * p.min_area_px:
            keep[lab == r.label] = True
    keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    return keep, ("dark" if dark else "light")


# ----------------------------------------------------------------------------- skeleton -> graph
def _neighbours(skel):
    return convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - skel


def _trace(start, prev, skel_set, stop):
    """Follow skeleton pixels from `start` (coming from `prev`) until a stop pixel or a dead end."""
    path = [start]
    cur, last = start, prev
    while cur not in stop or len(path) == 1:
        nxt = [(cur[0] + dy, cur[1] + dx) for dy, dx in NB8 if (cur[0] + dy, cur[1] + dx) in skel_set]
        nxt = [q for q in nxt if q != last and q not in path[-3:]]
        if not nxt:
            break
        # prefer a stop pixel, then straight neighbours (avoids cutting corners twice)
        nxt.sort(key=lambda q: (q not in stop, abs(q[0] - cur[0]) + abs(q[1] - cur[1])))
        last, cur = cur, nxt[0]
        path.append(cur)
        if cur in stop:
            break
    return path


def build_graph(skel):
    """networkx MultiGraph from a 1-pixel skeleton.

    Junction pixels (3+ neighbours) that touch each other form one node; crack ends
    (1 neighbour) are nodes too. Edges keep their ordered pixel path (row, col)."""
    skel = skel.astype(bool)
    nb = _neighbours(skel)
    junction_px = skel & (nb >= 3)
    jlab = measure.label(junction_px, connectivity=2)
    G = nx.MultiGraph()
    node_of = {}
    for r in measure.regionprops(jlab):
        nid = len(G)
        G.add_node(nid, pos=(float(r.centroid[0]), float(r.centroid[1])), kind="junction")
        for y, x in r.coords:
            node_of[(y, x)] = nid
    for y, x in zip(*np.nonzero(skel & (nb == 1))):
        nid = len(G)
        G.add_node(nid, pos=(float(y), float(x)), kind="end")
        node_of[(y, x)] = nid
    skel_set = set(zip(*np.nonzero(skel)))
    stop = set(node_of)
    seen = set()
    for (y, x), nid in node_of.items():
        for dy, dx in NB8:
            q = (y + dy, x + dx)
            if q not in skel_set or node_of.get(q) == nid:
                continue
            if q in stop:                                          # two nodes directly adjacent
                key = tuple(sorted([(y, x), q]))
                if key not in seen and node_of[q] != nid:
                    seen.add(key)
                    G.add_edge(nid, node_of[q], path=np.array([(y, x), q]))
                continue
            if (q, (y, x)) in seen:
                continue
            path = _trace(q, (y, x), skel_set, stop)
            end = path[-1]
            if len(path) > 1:
                seen.add((path[-2], end))
            seen.add((q, (y, x)))
            full = [(y, x)] + path
            if end in stop:
                key = (min(full[1], full[-2]), max(full[1], full[-2]), len(full))
                if key in seen:
                    continue
                seen.add(key)
                G.add_edge(nid, node_of[end], path=np.array(full))
    # closed loops without any junction (e.g. a ring-shaped crack)
    used = np.zeros_like(skel)
    for *_, d in G.edges(data=True):
        used[d["path"][:, 0], d["path"][:, 1]] = True
    for r in measure.regionprops(measure.label(skel & ~used & (nb == 2), connectivity=2)):
        if r.area >= 10:
            y, x = r.coords[0]
            nid = len(G)
            G.add_node(nid, pos=(float(y), float(x)), kind="loop")
            path = _trace(tuple(r.coords[1]) if len(r.coords) > 1 else (y, x), (y, x),
                          set(map(tuple, r.coords)), {(y, x)})
            G.add_edge(nid, nid, path=np.array([(y, x)] + path))
    return G


def prune(skel, min_len, half_width, rounds=3):
    """Remove dead-end branches shorter than min_len plus the crack width where they start.

    Every skeleton of a wide crack has small side branches ("spurs") that end at bumps of
    the crack edge; a real crack branch is clearly longer than the crack is wide."""
    skel = skel.copy()
    for _ in range(rounds):
        G = build_graph(skel)
        removed = 0
        for u, v, d in list(G.edges(data=True)):
            deg_u, deg_v = G.degree(u), G.degree(v)
            hub = v if deg_u == 1 else u
            y, x = (int(round(c)) for c in G.nodes[hub]["pos"])
            limit = min_len + 2 * half_width[y, x]
            if (deg_u == 1) != (deg_v == 1) and len(d["path"]) < limit:
                inner = d["path"][:-1] if deg_u == 1 else d["path"][1:]
                skel[inner[:, 0], inner[:, 1]] = False
                removed += 1
        skel = skeletonize(skel)
        if removed == 0:
            break
    return skel


def bridge(skel, max_gap):
    """Extend each crack end along its direction; if it reaches another crack within
    max_gap pixels, draw the missing piece (faint crack sections often break up)."""
    G = build_graph(skel)
    H, W = skel.shape
    out = skel.copy()
    for n, d in G.nodes(data=True):
        if d["kind"] != "end" or G.degree(n) != 1:
            continue
        (_, _, e), = [(u, v, ed) for u, v, ed in G.edges(n, data=True)]
        path = e["path"]
        if len(path) < 6:
            continue
        here = np.array(d["pos"])
        if np.linalg.norm(path[0] - here) > np.linalg.norm(path[-1] - here):
            path = path[::-1]                                    # path now starts at the end node
        k = min(len(path) - 1, 10)
        direc = (path[0] - path[k]).astype(float)
        direc /= np.linalg.norm(direc) + 1e-9
        own = set(map(tuple, path[:25]))
        for t in np.arange(2, max_gap, 0.5):
            q = np.round(here + t * direc).astype(int)
            if not (0 <= q[0] < H and 0 <= q[1] < W):
                break
            hit = [tuple(q + o) for o in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
                   if 0 <= q[0] + o[0] < H and 0 <= q[1] + o[1] < W and skel[q[0] + o[0], q[1] + o[1]]
                   and tuple(q + o) not in own]
            if hit:
                cv2.line(out.view(np.uint8), (int(here[1]), int(here[0])), (hit[0][1], hit[0][0]), 1, 1)
                break
    return skeletonize(out)


def merge_close_junctions(G, half_width):
    """Contract very short edges between two junctions into one node.

    Where wide cracks meet, the skeleton often splits one junction into two close ones
    joined by a stub shorter than the crack width (a Y would turn into two T's)."""
    changed = True
    while changed:
        changed = False
        for u, v, k, d in list(G.edges(keys=True, data=True)):
            if u == v or G.nodes[u]["kind"] != "junction" or G.nodes[v]["kind"] != "junction":
                continue
            pu, pv = np.array(G.nodes[u]["pos"]), np.array(G.nodes[v]["pos"])
            w = 2 * max(half_width[int(pu[0]), int(pu[1])], half_width[int(pv[0]), int(pv[1])])
            if len(d["path"]) <= max(4, 1.0 * w):
                G.remove_edge(u, v, k)
                new = (pu + pv) / 2
                for _, nbr, ed in list(G.edges(v, data=True)):
                    G.add_edge(u, nbr if nbr != v else u, path=ed["path"])
                G.remove_node(v)
                G.nodes[u]["pos"] = (float(new[0]), float(new[1]))
                changed = True
                break
    return nx.convert_node_labels_to_integers(G)


def skeleton_graph(mask, p: Params):
    half_width = distance_transform_edt(mask)
    skel = skeletonize(mask)
    skel = prune(skel, p.prune_px, half_width)
    if p.bridge_px > 0:
        skel = bridge(skel, p.bridge_px)
        skel = prune(skel, p.prune_px, half_width, rounds=1)
    return skel, merge_close_junctions(build_graph(skel), half_width)


# ----------------------------------------------------------------------------- measurements
def path_length(path, sigma=2.0):
    """Length of a pixel path after light smoothing (removes the pixel staircase)."""
    if len(path) < 3:
        return float(np.linalg.norm(path[-1] - path[0])) if len(path) == 2 else 0.0
    sm = gaussian_filter1d(path.astype(float), sigma, axis=0, mode="nearest")
    sm[0], sm[-1] = path[0], path[-1]
    return float(np.linalg.norm(np.diff(sm, axis=0), axis=1).sum())


def _branch_direction(G, node, path, reach):
    """Unit vector (dy, dx) from a node into one of its edges, `reach` pixels along the edge."""
    here = np.array(G.nodes[node]["pos"])
    if np.linalg.norm(path[0] - here) > np.linalg.norm(path[-1] - here):
        path = path[::-1]
    k = min(len(path) - 1, int(reach))
    v = path[k] - here
    return v / (np.linalg.norm(v) + 1e-9)


def junction_type(angles_deg):
    """T: one angle near 180 (a crack ran into an older one); Y: all near 120; X: 4+ branches."""
    if len(angles_deg) >= 4:
        return "X"
    return "T" if max(angles_deg) >= 150 else "Y"


@dataclass
class Network:
    mask: np.ndarray
    skeleton: np.ndarray
    graph: nx.MultiGraph
    cells: np.ndarray                 # label image of closed cells (0 = crack or open border region)
    polarity: str
    px_per_mm: float = None
    edges: list = field(default_factory=list)
    junctions: list = field(default_factory=list)
    cell_stats: list = field(default_factory=list)

    @property
    def unit(self):
        return "mm" if self.px_per_mm else "px"

    def L(self, v):                   # length in output units
        return v / self.px_per_mm if self.px_per_mm else v

    def A(self, v):                   # area in output units
        return v / self.px_per_mm ** 2 if self.px_per_mm else v

    def summary(self):
        H, W = self.mask.shape
        u = self.unit
        total = sum(e["length"] for e in self.edges)
        kinds = [j["type"] for j in self.junctions]
        n_j = len(kinds)
        ends = sum(1 for _, d in self.graph.nodes(data=True) if d["kind"] == "end"
                   and not _at_border(d["pos"], H, W))
        th = np.deg2rad([e["orientation_deg"] for e in self.edges])
        wts = np.array([e["length"] for e in self.edges])
        aniso = float(np.abs((wts * np.exp(2j * th)).sum()) / max(wts.sum(), 1e-9)) if len(th) else 0.0
        dom = float(np.rad2deg(np.angle((wts * np.exp(2j * th)).sum()) / 2) % 180) if len(th) else 0.0
        areas = [c["area"] for c in self.cell_stats]
        sides = [c["corners"] for c in self.cell_stats]
        return {
            "polarity": self.polarity,
            f"total_crack_length_{u}": round(self.L(total), 1),
            f"crack_density_{u}_per_{u}2": round(self.L(total) / self.A(H * W), 4),
            "crack_area_%": round(100 * float(self.mask.mean()), 2),
            "segments": len(self.edges),
            f"median_segment_length_{u}": round(self.L(float(np.median([e["length"] for e in self.edges]))), 1)
            if self.edges else 0,
            f"median_width_{u}": round(self.L(float(np.median([e["width"] for e in self.edges]))), 2)
            if self.edges else 0,
            "junctions": n_j,
            "T_junctions": kinds.count("T"),
            "Y_junctions": kinds.count("Y"),
            "X_junctions": kinds.count("X"),
            "T_fraction": round(kinds.count("T") / max(kinds.count("T") + kinds.count("Y"), 1), 2),
            "free_crack_ends": ends,
            "closed_cells": len(self.cell_stats),
            f"median_cell_area_{u}2": round(self.A(float(np.median(areas))), 1) if areas else 0,
            "mean_cell_corners": round(float(np.mean(sides)), 2) if sides else 0,
            "orientation_anisotropy": round(aniso, 2),
            "dominant_orientation_deg": round(dom, 0),
        }


def _at_border(pos, H, W, m=3):
    y, x = pos
    return y < m or x < m or y > H - 1 - m or x > W - 1 - m


def measure_network(mask, skel, G, polarity, px_per_mm=None):
    H, W = mask.shape
    half_width = distance_transform_edt(mask)
    net = Network(mask, skel, G, None, polarity, px_per_mm)
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        path = d["path"]
        length = path_length(path)
        chord = float(np.linalg.norm(path[-1] - path[0]))
        dy, dx = path[-1] - path[0]
        ang = float(np.rad2deg(np.arctan2(-dy, dx)) % 180)                  # 0 = horizontal, 90 = vertical
        width = float(np.median(2 * half_width[path[:, 0], path[:, 1]]))
        d.update(length=length, width=width)
        net.edges.append({"id": i, "u": u, "v": v, "length": length, "chord": chord,
                          "tortuosity": length / chord if chord > 0 else float("inf"),
                          "orientation_deg": ang, "width": width})
    # closed cells: background regions fully surrounded by cracks
    lab = measure.label(~skel, connectivity=1)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    corners = {}                                        # cell label -> number of corners
    for n, nd in G.nodes(data=True):
        deg = G.degree(n)
        if deg < 3:
            continue
        here = np.array(nd["pos"])
        reach = max(6.0, 2.0 * float(half_width[int(here[0]), int(here[1])]) + 4)
        dirs = [_branch_direction(G, n, ed["path"], reach) for _, _, ed in G.edges(n, data=True)]
        ang = np.sort(np.arctan2([-v[0] for v in dirs], [v[1] for v in dirs]))
        gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
        # each angle between two neighbouring branches belongs to the cell that fills it;
        # it is a corner of that cell unless the boundary runs straight through (~180 deg)
        for a0, g in zip(ang, gaps):
            if np.rad2deg(g) < 155:
                mid = a0 + g / 2
                q = np.round(here + 1.5 * reach * np.array([-np.sin(mid), np.cos(mid)])).astype(int)
                if 0 <= q[0] < H and 0 <= q[1] < W and lab[q[0], q[1]]:
                    corners[lab[q[0], q[1]]] = corners.get(lab[q[0], q[1]], 0) + 1
        if _at_border(here, H, W, 5):
            continue
        gaps = np.rad2deg(gaps)
        nd["type"] = junction_type(gaps)
        net.junctions.append({"node": n, "y": here[0], "x": here[1], "degree": deg,
                              "angles_deg": [round(float(g), 1) for g in sorted(gaps)], "type": nd["type"]})
    cells = np.zeros_like(lab)
    k = 0
    for r in measure.regionprops(lab):
        if r.label in border or r.area < 30:
            continue
        k += 1
        cells[lab == r.label] = k
        net.cell_stats.append({"id": k, "area": float(r.area), "corners": corners.get(r.label, 0),
                               "eq_diameter": float(r.equivalent_diameter_area),
                               "elongation": float(r.axis_major_length / max(r.axis_minor_length, 1))})
    net.cells = cells
    return net


def analyse(rgb, p: Params = Params()):
    mask, polarity = segment(rgb, p)
    skel, G = skeleton_graph(mask, p)
    return measure_network(mask, skel, G, polarity, p.px_per_mm)
