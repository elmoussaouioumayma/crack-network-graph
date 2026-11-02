"""Example images of crack networks with a known ground-truth network.

Three surfaces, all generated here (no external images):
    mud    - dried mud: brown plates, wide dark cracks, first cracks widest
    glaze  - crazed ceramic glaze: smooth coloured tile, thin dark hairline cracks, bubbles
    paint  - old paint on wood: the cracks show the LIGHTER wood underneath

Two ways a crack network can form, which leave different junctions:
    sequential - cracks form one after another; each new crack runs into an older one
                 at about 90 degrees -> T junctions (typical for drying mud and paint)
    voronoi    - cracks form at the same time and meet at about 120 degrees -> Y junctions
                 (typical for mature networks, columnar basalt)
    mixed      - a Y-type network that is later subdivided by T-type cracks

    python synth.py                   # writes examples/*.png and their truth files
"""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import Voronoi
from shapely.geometry import LineString, Polygon, box
from shapely.ops import split, unary_union
from skimage import measure


def _noise(shape, sigma, rng):
    n = gaussian_filter(rng.normal(0, 1, shape), sigma)
    return n / (n.std() + 1e-9)


# ----------------------------------------------------------------------------- geometry
def sequential_network(W, H, mean_cell, rng, start=None):
    """Hierarchical fracture: repeatedly split the largest piece across its long axis.

    Returns cracks as (LineString, generation) and the junctions they create (all T)."""
    pieces = start or [box(-5, -5, W + 5, H + 5)]
    cracks, juncs = [], []
    target = mean_cell ** 2
    gen = 0
    while True:
        areas = np.array([p.area for p in pieces])
        big = np.nonzero(areas > target * rng.uniform(1.3, 2.2, len(areas)))[0]
        if len(big) == 0:
            break
        i = big[np.argmax(areas[big])]
        p = pieces.pop(i)
        xy = np.asarray(p.exterior.coords)[:-1]
        c = np.asarray(p.centroid.coords[0]) + rng.normal(0, 0.12 * np.sqrt(p.area), 2)
        cov = np.cov((xy - xy.mean(0)).T)
        long_axis = np.linalg.eigh(cov)[1][:, -1]
        ang = np.arctan2(long_axis[1], long_axis[0]) + np.pi / 2 + rng.normal(0, 0.2)
        d = np.array([np.cos(ang), np.sin(ang)]) * 4 * (W + H)
        cut = LineString([c - d, c + d]).intersection(p)
        if cut.is_empty or cut.geom_type != "LineString" or cut.length < 5:
            pieces.append(p)
            break
        parts = split(p, cut)
        if len(parts.geoms) < 2:
            pieces.append(p)
            continue
        pieces.extend(parts.geoms)
        cracks.append((cut, gen))
        gen += 1
        for e in cut.boundary.geoms:
            juncs.append((e.x, e.y, "T"))
    return cracks, juncs, pieces


def voronoi_network(W, H, mean_cell, rng, relax=2):
    """Voronoi cells (with Lloyd relaxation): cracks meet in Y junctions."""
    n = int(W * H / mean_cell ** 2 * 1.15)
    pts = rng.uniform([-60, -60], [W + 60, H + 60], (n, 2))
    frame = box(-60, -60, W + 60, H + 60)
    for _ in range(relax + 1):
        vor = Voronoi(pts)
        cells = []
        for r in vor.point_region:
            reg = vor.regions[r]
            if -1 in reg or len(reg) < 3:
                cells.append(None)
                continue
            cells.append(Polygon(vor.vertices[reg]).intersection(frame))
        new = [np.asarray(c.centroid.coords[0]) if c is not None and not c.is_empty else p
               for c, p in zip(cells, pts)]
        pts = np.array(new)
    cracks, juncs = [], []
    for (a, b) in vor.ridge_vertices:
        if a >= 0 and b >= 0:
            cracks.append((LineString([vor.vertices[a], vor.vertices[b]]), 0))
    for v in vor.vertices:
        juncs.append((v[0], v[1], "Y"))
    pieces = [c for c in cells if c is not None and not c.is_empty and c.geom_type == "Polygon"]
    return cracks, juncs, pieces


def warp(cracks, juncs, W, H, rng, amp=(10.0, 2.5), scale=(60.0, 12.0)):
    """Bend all cracks with one smooth displacement field (keeps every junction connected)."""
    gx = sum(_noise((H + 200, W + 200), s, rng) * a for a, s in zip(amp, scale))
    gy = sum(_noise((H + 200, W + 200), s, rng) * a for a, s in zip(amp, scale))

    def f(x, y):
        xi, yi = int(np.clip(round(x) + 100, 0, W + 199)), int(np.clip(round(y) + 100, 0, H + 199))
        return x + gx[yi, xi], y + gy[yi, xi]

    out = []
    for line, g in cracks:
        dense = [line.interpolate(t) for t in np.arange(0, line.length, 1.5)] + [line.interpolate(line.length)]
        out.append((LineString([f(p.x, p.y) for p in dense]), g))
    return out, [(*f(x, y), t) for x, y, t in juncs]


def truth_from_network(cracks, juncs, W, H, margin=3):
    """Ground truth inside the image: crack length, junctions by type, closed cells."""
    frame = box(0, 0, W - 1, H - 1)
    lines = unary_union([ln for ln, _ in cracks]).intersection(frame)
    inner = box(margin, margin, W - 1 - margin, H - 1 - margin)
    j_in = [(x, y, t) for x, y, t in juncs if inner.contains(Polygon([(x - .1, y - .1), (x + .1, y - .1),
                                                                        (x + .1, y + .1), (x - .1, y + .1)]))]
    # closed cells: regions fully enclosed by cracks (raster, so tiny gaps at junctions do not matter)
    thin = np.zeros((H, W), np.uint8)
    for ln, _ in cracks:
        cv2.polylines(thin, [np.asarray(ln.coords).round().astype(np.int32)], False, 1, 3)
    lab = measure.label(thin == 0, connectivity=1)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    cells = [r for r in measure.regionprops(lab) if r.label not in border and r.label != 0]
    return {"total_length_px": float(lines.length),
            "junctions": len(j_in),
            "T_junctions": sum(t == "T" for *_, t in j_in),
            "Y_junctions": sum(t == "Y" for *_, t in j_in),
            "closed_cells": len(cells),
            "median_cell_area_px": float(np.median([c.area for c in cells])) if cells else 0.0,
            "junction_xy_type": [(round(x, 1), round(y, 1), t) for x, y, t in j_in]}


def make_network(kind, W, H, mean_cell, rng):
    if kind == "sequential":
        cracks, juncs, _ = sequential_network(W, H, mean_cell, rng)
    elif kind == "voronoi":
        cracks, juncs, _ = voronoi_network(W, H, mean_cell, rng)
    else:                                                    # mixed
        cracks, juncs, pieces = voronoi_network(W, H, mean_cell * 1.9, rng)
        c2, j2, _ = sequential_network(W, H, mean_cell, rng, start=pieces)
        cracks += [(ln, g + 1) for ln, g in c2]
        juncs += j2
    cracks, juncs = warp(cracks, juncs, W, H, rng)
    return cracks, juncs


# ----------------------------------------------------------------------------- rendering
def crack_field(cracks, W, H, width_fn, rng, rough=0.6):
    """Crack mask (float 0..1) with rough edges; width_fn(generation) gives the width in px."""
    m = np.zeros((H, W), np.float32)
    for line, g in cracks:
        pts = np.asarray(line.coords).round().astype(np.int32)
        w = max(1, int(round(width_fn(g))))
        cv2.polylines(m, [pts], False, 1.0, w, cv2.LINE_AA)
    m = gaussian_filter(m, 0.6) + rough * 0.15 * _noise((H, W), 1.0, rng)
    return np.clip(m * 1.4, 0, 1)


def render(surface, cracks, W, H, rng):
    yy, xx = np.mgrid[0:H, 0:W] / max(W, H)
    if surface == "mud":
        n = max(g for _, g in cracks) + 1
        field = crack_field(cracks, W, H, lambda g: 9 - 6 * g / max(n - 1, 1) + rng.normal(0, 0.6), rng)
        d = distance_transform_edt(field < 0.5)
        curl = np.exp(-d / 6.0)                                   # plate edges curl up and look darker
        base = np.array([150, 112, 78]) / 255 * (1 + 0.10 * _noise((H, W), 25, rng)[..., None]
                                                + 0.05 * _noise((H, W), 2, rng)[..., None])
        img = base * (1 - 0.25 * curl[..., None]) * (1 - 0.12 * xx[..., None])
        crack_col = np.array([45, 30, 22]) / 255 * (1 + 0.2 * _noise((H, W), 1.5, rng)[..., None])
        img = img * (1 - field[..., None]) + crack_col * field[..., None]
        for _ in range(250):                                      # small pebbles / specks
            cv2.circle(img, (int(rng.uniform(0, W)), int(rng.uniform(0, H))), int(rng.uniform(1, 3)),
                       tuple(float(v) for v in np.array([110, 85, 60]) / 255 * rng.uniform(0.8, 1.2)), -1)
    elif surface == "glaze":
        field = crack_field(cracks, W, H, lambda g: 1.6 if g < 3 else 1.2, rng, rough=0.3)
        base = np.array([96, 168, 160]) / 255                     # celadon green-blue
        light = 1 + 0.12 * np.exp(-((xx - 0.3) ** 2 + (yy - 0.25) ** 2) / 0.05) - 0.08 * yy
        img = base * light[..., None] * (1 + 0.025 * _noise((H, W), 8, rng)[..., None])
        stain = np.array([40, 60, 58]) / 255
        img = img * (1 - 0.85 * field[..., None]) + stain * 0.85 * field[..., None]
        for _ in range(35):                                       # glaze bubbles (round, not cracks)
            c = (int(rng.uniform(0, W)), int(rng.uniform(0, H)))
            r = int(rng.uniform(2, 5))
            cv2.circle(img, c, r, tuple(float(v) for v in base * 0.55), -1, cv2.LINE_AA)
            cv2.circle(img, (c[0] - 1, c[1] - 1), max(1, r // 3), (0.85, 0.92, 0.9), -1, cv2.LINE_AA)
    else:                                                         # paint on wood
        field = crack_field(cracks, W, H, lambda g: 4.5 - 2.5 * min(g, 30) / 30 + rng.normal(0, 0.5), rng)
        paint = np.array([168, 38, 36]) / 255 * (1 + 0.06 * _noise((H, W), 3, rng)[..., None]
                                                + 0.05 * _noise((H, W // 3), 1, rng).repeat(3, 1)[:, :W, None])
        grain = np.sin(yy * 260 + 4 * _noise((H, W), 20, rng)) * 0.5 + 0.5
        wood = np.array([214, 178, 120]) / 255 * (0.85 + 0.15 * grain[..., None])
        img = paint * (1 - field[..., None]) + wood * field[..., None]
        d = distance_transform_edt(field < 0.5)
        img *= (1 - 0.18 * np.exp(-d / 2.0))[..., None]          # thin shadow at the paint edge
    img = img + rng.normal(0, 0.012, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


EXAMPLES = [  # name, surface, network, mean cell size (px), seed
    ("dried_mud", "mud", "sequential", 120, 1),
    ("glaze_crazing", "glaze", "mixed", 70, 2),
    ("old_paint", "paint", "sequential", 70, 3),
    ("basalt_like_mud", "mud", "voronoi", 110, 4),
]


def make_example(surface, network, mean_cell, seed, W=900, H=640):
    rng = np.random.default_rng(seed)
    cracks, juncs = make_network(network, W, H, mean_cell, rng)
    img = render(surface, cracks, W, H, rng)
    return img, truth_from_network(cracks, juncs, W, H)


def main():
    out = Path("examples")
    out.mkdir(exist_ok=True)
    for name, surface, network, cell, seed in EXAMPLES:
        img, truth = make_example(surface, network, cell, seed)
        cv2.imwrite(str(out / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        (out / f"{name}.truth.json").write_text(json.dumps({"surface": surface, "network": network, **truth}))
        print(f"{name}: {truth['junctions']} junctions (T {truth['T_junctions']}, Y {truth['Y_junctions']}), "
              f"{truth['closed_cells']} cells")


if __name__ == "__main__":
    main()
