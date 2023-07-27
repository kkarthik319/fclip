#!/usr/bin/env python
"""Process Huang's wireframe dataset for L-CNN network
Usage:
    dataset/wireframe.py <src> <dst>
    dataset/wireframe.py (-h | --help )

Examples:
    python dataset/wireframe.py /datadir/wireframe Good/wireframe

Arguments:
    <src>                Original Good directory of Huang's wireframe dataset
    <dst>                Directory of the output

Options:
   -h --help             Show this screen.
"""

import os
import sys
import json
from itertools import combinations

import cv2
import numpy as np
import skimage.draw
from scipy.ndimage import zoom
from tqdm import tqdm

try:
    sys.path.append(".")
    sys.path.append("..")
    from FClip.utils import parmap
except Exception:
    raise


def inrange(v, shape):
    return 0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]


def to_int(x):
    return tuple(map(int, x))


def save_heatmap(prefix, image, lines):
    im_rescale = (512, 512)
    heatmap_scale = (128, 128)

    fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]
    jmap = np.zeros((1,) + heatmap_scale, dtype=np.float32)  # (1, 128, 128)
    joff = np.zeros((1, 2) + heatmap_scale, dtype=np.float32)  # (1, 2, 128, 128)
    lmap = np.zeros(heatmap_scale, dtype=np.float32)  # (128, 128)

    # the coordinate of lines can not equal to 128 (less than 128).
    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)
    lines = lines[:, :, ::-1]  # change position of x and y

    junc = []
    jids = {}

    # collecting junction endpoints (jun) and number them in dictionary (junc, jids).
    def jid(jun):
        jun = tuple(jun[:2])
        if jun in jids:
            return jids[jun]
        jids[jun] = len(junc)
        junc.append(np.array(jun + (0,)))
        return len(junc) - 1

    lnid = []
    lpos, lneg = [], []
    # drawing the heat map of line.
    for v0, v1 in lines:
        lnid.append((jid(v0), jid(v1)))
        lpos.append([junc[jid(v0)], junc[jid(v1)]])

        vint0, vint1 = to_int(v0), to_int(v1)
        jmap[0][vint0] = 1
        jmap[0][vint1] = 1
        rr, cc, value = skimage.draw.line_aa(*to_int(v0), *to_int(v1))
        lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

    for v in junc:
        vint = to_int(v[:2])
        joff[0, :, vint[0], vint[1]] = v[:2] - vint - 0.5

    llmap = zoom(lmap, [0.5, 0.5])  # down sampler lmap
    lineset = set([frozenset(l) for l in lnid])
    for i0, i1 in combinations(range(len(junc)), 2):
        if frozenset([i0, i1]) not in lineset:
            v0, v1 = junc[i0], junc[i1]
            vint0, vint1 = to_int(v0[:2] / 2), to_int(v1[:2] / 2)  #
            rr, cc, value = skimage.draw.line_aa(*vint0, *vint1)
            lneg.append([v0, v1, i0, i1, np.average(np.minimum(value, llmap[rr, cc]))])  # ?why minimum  hardness score?

    assert len(lneg) != 0
    lneg.sort(key=lambda l: -l[-1])

    junc = np.array(junc, dtype=np.float32)
    Lpos = np.array(lnid, dtype=np.int32)
    Lneg = np.array([l[2:4] for l in lneg][:4000], dtype=np.int32)
    lpos = np.array(lpos, dtype=np.float32)
    lneg = np.array([l[:2] for l in lneg[:2000]], dtype=np.float32)

    image = cv2.resize(image, im_rescale)

    np.savez_compressed(
        f"{prefix}_label.npz",
        aspect_ratio=image.shape[1] / image.shape[0],
        jmap=jmap,  # [J, H, W]    Junction heat map
        joff=joff,  # [J, 2, H, W] Junction offset within each pixel
        lmap=lmap,  # [H, W]       Line heat map with anti-aliasing
        junc=junc,  # [Na, 3]      Junction coordinate
        Lpos=Lpos,  # [M, 2]       Positive lines represented with junction indices
        Lneg=Lneg,  # [M, 2]       Negative lines represented with junction indices
        lpos=lpos,  # [Np, 2, 3]   Positive lines represented with junction coordinates
        lneg=lneg,  # [Nn, 2, 3]   Negative lines represented with junction coordinates
    )


def main(data_root):
    for batch in ["test", "train"]:  # "train", "valid"
        anno_file = os.path.join(data_root, f"{batch}", "annotations", "annotations.json")
        with open(anno_file, "r") as f:
            dataset = json.load(f)
        only_need_files = set(os.listdir(f"/Users/saikrishnaseelamlakshmi/Downloads/{batch}"))
        print("total files", len(only_need_files))
        def handle(data):
            if data['filename'] not in only_need_files:
                return
            im = cv2.imread(os.path.join(data_root, "images", data["filename"]))
            prefix = data["filename"]
            lines, done = [], set()
            for reg in data['regions']:
                xs, ys = reg['shape_attributes']['all_points_x'], reg['shape_attributes']['all_points_y']
                length = len(xs)
                for i in range(length):
                    j = (i + 1) % length
                    x1, y1, x2, y2 = xs[i], ys[i], xs[j], ys[j]
                    if f'{x1} {y1} {x2} {y2}' in done:
                        continue
                    lines.append([[xs[i], ys[i]], [xs[j], ys[j]]])
                    done.add(f'{x1} {y1} {x2} {y2}')
                    done.add(f'{x2} {y2} {x1} {y1}')
            lines0 = lines.copy()
            save_heatmap(f"./data/{batch}_gt/{prefix}", im[::, ::], np.array(lines0))

        # multiprocessing the function of handle with augment 'dataset'.
        for _, data in tqdm(dataset.items()):
            handle(data)


if __name__ == "__main__":
    main("../hawp/data/wireframe/")
