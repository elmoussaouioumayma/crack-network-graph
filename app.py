"""Live demo: drop a photo of a crack network, get it back as a measured graph.

    python app.py            # local, opens in the browser
Runs unchanged as a Hugging Face Space (SDK: gradio, app file: app.py).
"""

import json
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import cv2
import numpy as np

from network import Params, analyse
from viz import cells_image, charts, edges_csv, overlay, to_geojson, to_graphml

MAX_SIDE = 1400
EXAMPLES = sorted(str(p) for p in Path("examples").glob("*.png"))


def _dot(colour, text):
    return (f'<span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:{colour};'
            f'margin:0 4px 0 12px;vertical-align:middle"></span>{text}')


LEGEND = ("<div style='font-size:13px'>Junctions:" + _dot("#ff8c00", "T (≈ 90° / 180°)") + _dot("#1e90ff", "Y (≈ 120°)")
          + _dot("#ba55d3", "X (4 cracks)") + " &nbsp;&nbsp;&nbsp; Cell corners:"
          + "".join(_dot(c, k) for k, c in [("3", "#f4a259"), ("4", "#e9c46a"), ("5", "#8ab17d"), ("6", "#2a9d8f"),
                                             ("7+", "#264653")]) + "</div>")


def interpret(s):
    t, y = s["T_junctions"], s["Y_junctions"]
    if t + y < 5:
        return "Too few junctions for a statement about how the network formed."
    f = s["T_fraction"]
    corners = s["mean_cell_corners"]
    if f >= 0.65:
        story = ("Mostly **T junctions**: new cracks ran into older ones at about 90°, so the cracks formed "
                 "**one after another** (hierarchical cracking, e.g. one drying event).")
    elif f <= 0.35:
        story = ("Mostly **Y junctions** (about 120°): the cracks formed **at the same time**, or the network "
                 "has **matured** over repeated cycles (e.g. wetting and drying).")
    else:
        story = "A **mix of T and Y junctions**: an older network that was later divided by newer cracks."
    return (f"{story}\n\nT fraction {f:.2f} · {s['closed_cells']} closed cells with on average "
            f"{corners} corners · {s['free_crack_ends']} crack ends that stop in the open surface.")


def run(image, polarity, max_width, sensitivity, prune, bridge, px_per_mm):
    if image is None:
        raise gr.Error("Upload an image or click one of the examples.")
    rgb = np.asarray(image)[..., :3]
    scale = min(1.0, MAX_SIDE / max(rgb.shape[:2]))
    if scale < 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ppm = px_per_mm * scale if px_per_mm and px_per_mm > 0 else None
    p = Params(polarity=polarity, max_width_px=int(max_width), sensitivity=float(sensitivity),
               prune_px=float(prune), bridge_px=float(bridge), px_per_mm=ppm)
    net = analyse(rgb, p)
    s = net.summary()
    out = Path(tempfile.mkdtemp())
    (out / "crack_network.geojson").write_text(to_geojson(net))
    (out / "crack_segments.csv").write_text(edges_csv(net))
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    to_graphml(net, out / "crack_network.graphml")
    table = [[k.replace("_", " "), v] for k, v in s.items()]
    files = [str(out / f) for f in ("crack_network.geojson", "crack_segments.csv", "crack_network.graphml",
                                    "summary.json")]
    return overlay(rgb, net), cells_image(net), charts(net), table, interpret(s), files


with gr.Blocks(title="Crack network graph") as demo:
    gr.Markdown(
        "# Crack network → graph\n"
        "Drop a photo of a cracked surface (dried mud, glaze, old paint, concrete, a road ...). The cracks are "
        "turned into **vectors**: every crack segment and every junction becomes part of a graph, which is then "
        "measured: lengths, widths, directions, **T / Y / X junctions** and the closed **cells** between cracks.")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="numpy", label="Photo")
            with gr.Accordion("Settings", open=False):
                polarity = gr.Radio(["auto", "dark", "light"], value="auto",
                                    label="Cracks are ... than the surface (auto detects it)")
                max_width = gr.Slider(3, 40, value=15, step=1, label="Widest crack (px)")
                sensitivity = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Sensitivity (higher = fainter cracks)")
                prune = gr.Slider(0, 40, value=12, step=1, label="Remove dead-end branches shorter than (px)")
                bridge = gr.Slider(0, 40, value=15, step=1, label="Close gaps in cracks up to (px)")
                ppm = gr.Number(value=0, label="Scale: pixels per mm (0 = report in pixels)")
            btn = gr.Button("Analyse", variant="primary")
            gr.Examples(EXAMPLES, inputs=inp, label="Examples (generated images)")
        with gr.Column(scale=2):
            with gr.Row():
                out_graph = gr.Image(label="Graph (vectors) with junction types")
                out_cells = gr.Image(label="Closed cells by number of corners")
            gr.HTML(LEGEND)
            story = gr.Markdown()
            out_plot = gr.Plot(show_label=False)
            with gr.Row():
                table = gr.Dataframe(headers=["measure", "value"], label="Summary", wrap=True)
                files = gr.File(label="Downloads: GeoJSON (vectors), CSV, GraphML, JSON", file_count="multiple")
    args = [inp, polarity, max_width, sensitivity, prune, bridge, ppm]
    outs = [out_graph, out_cells, out_plot, table, story, files]
    btn.click(run, args, outs)
    inp.upload(run, args, outs)

if __name__ == "__main__":
    demo.launch()
