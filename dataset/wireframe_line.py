#!/usr/bin/env python
"""Process Huang's wireframe dataset for L-CNN network
Usage:
    dataset/wireframe.py <src> <dst>
    dataset/wireframe.py (-h | --help )

Examples:
    python dataset/wireframe.py /datadir/wireframe Good/wireframe
    python dataset/wireframe_line.py /home/dxl/Data/wireframe_raw

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
import matplotlib.pyplot as plt
from docopt import docopt
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

    lcmap = np.zeros(heatmap_scale, dtype=np.float32)  # (128, 128)
    lcoff = np.zeros((2,) + heatmap_scale, dtype=np.float32)  # (2, 128, 128)
    lleng = np.zeros(heatmap_scale, dtype=np.float32)  # (128, 128)
    angle = np.zeros(heatmap_scale, dtype=np.float32)  # (128, 128)

    # the coordinate of lines can not equal to 128 (less than 128).
    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)
    lines = lines[:, :, ::-1]  # change position of x and y --> (r, c)

    for v0, v1 in lines:
        v = (v0 + v1) / 2
        vint = to_int(v)
        lcmap[vint] = 1
        lcoff[:, vint[0], vint[1]] = v - vint - 0.5
        lleng[vint] = np.sqrt(np.sum((v0 - v1) ** 2)) / 2  # L

        if v0[0] <= v[0]:
            vv = v0
        else:
            vv = v1

        # the angle under the image coordinate system (r, c)
        # theta means the component along the c direction on the unit vector
        if np.sqrt(np.sum((vv - v) ** 2)) <= 1e-4:
            continue
        angle[vint] = np.sum((vv - v) * np.array([0., 1.])) / np.sqrt(np.sum((vv - v) ** 2))  # theta
    image = cv2.resize(image, im_rescale)

    np.savez_compressed(
        f"{prefix}_line.npz",
        # aspect_ratio=image.shape[1] / image.shape[0],
        lcmap=lcmap,
        lcoff=lcoff,
        lleng=lleng,
        angle=angle,
    )
    # cv2.imwrite(f"{prefix}.png", image)


def coor_rot90(coordinates, center, k):

    # !!!rotate the coordinates 90 degree anticlockwise on image!!!!

    # (x, y) --> (p-q+y, p+q-x) means point (x,y) rotate 90 degree clockwise along center (p,q)
    # but, the y direction of coordinates is inverse, not up but down.
    # so it equals to rotate the coordinate anticlockwise.

    # coordinares: [n, 2]; center: (p, q) rotation center.
    # coordinates and center should follow the (x, y) order, not (h, w).
    new_coor = coordinates.copy()
    p, q = center
    for i in range(k):
        x = p - q + new_coor[:, 1:2]
        y = p + q - new_coor[:, 0:1]
        new_coor = np.concatenate([x, y], 1)
    return new_coor


def prepare_rotation(image, lines):
    heatmap_scale = (512, 512)

    fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]

    # the coordinate of lines can not equal to 128 (less than 128).
    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)

    im = cv2.resize(image, heatmap_scale)

    return im, lines


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

        for _, data in tqdm(dataset.items()):
            handle(data)


if __name__ == "__main__":
    main("../hawp/data/wireframe/")

