"""Pipeline figure for the README (the app screenshots are taken from the running app).

    python make_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

from network import Params, analyse
from viz import cells_image, overlay


def main():
    rgb = cv2.cvtColor(cv2.imread("examples/glaze_crazing.png"), cv2.COLOR_BGR2RGB)
    net = analyse(rgb, Params())
    y0, y1, x0, x1 = 120, 420, 250, 650
    raw = skeletonize(net.mask)
    panels = [(rgb, "1. Photo"),
              (net.mask, "2. Crack mask\n(black-hat + hysteresis)"),
              (raw, "3. Skeleton\n(1 px centre lines, with spurs)"),
              (overlay(rgb, net), "4. Graph\n(pruned, gaps closed, junctions typed)"),
              (cells_image(net), "5. Closed cells\n(coloured by corners)")]
    fig, ax = plt.subplots(1, 5, figsize=(20, 3.9))
    for a, (im, t) in zip(ax, panels):
        a.imshow(im[y0:y1, x0:x1], cmap="gray" if im.ndim == 2 else None)
        a.set_title(t, fontsize=11)
        a.axis("off")
    fig.tight_layout()
    fig.savefig("images/pipeline.png", dpi=100)

    fig, ax = plt.subplots(1, 4, figsize=(20, 3.6))
    for a, n in zip(ax, ["dried_mud", "basalt_like_mud", "glaze_crazing", "old_paint"]):
        a.imshow(cv2.cvtColor(cv2.imread(f"examples/{n}.png"), cv2.COLOR_BGR2RGB))
        a.set_title(n.replace("_", " "), fontsize=11)
        a.axis("off")
    fig.tight_layout()
    fig.savefig("images/examples.png", dpi=80)
    print("Figures written to images/")


if __name__ == "__main__":
    main()
