#!/usr/bin/env python3
"""Evaluate sAP5, sAP10, sAP15 for LCNN
Usage:
    eval-sAP.py [options] <path>...
    eval-sAP.py (-h | --help )

Arguments:
    <path>                           One or more directories from train.py

Options:
   -h --help                         Show this screen.
   -c --config <config>              expe config   [default: config]
   -m --mode <mode>                  Good set name [default: shanghaiTech]
"""

import os
import glob
import numpy as np
from docopt import docopt

import FClip.utils
import FClip.metric
from FClip.config import C
from FClip.line_parsing import line_parsing_from_npz
from tqdm import tqdm

# python eval-sAP.py -m shanghaiTech path/to/npz/directories


def line_center_score(path, GT, threshold=5, confidence_threshold=0.675, csv_file_path="temp.txt"):
    preds = sorted(glob.glob(path))
    gts = sorted(glob.glob(GT))
    n_gt = 0
    n_pt = 0
    tps, fps, scores, f1_score, precision, recall = [], [], [], [], [], []
    csv_file = open(csv_file_path, "w")
    csv_file.write(f"Precision, Recall, F1_score\n")

    for pred_name, gt_name in tqdm(zip(preds, gts)):
        #print(pred_name,gt_name)
        with np.load(gt_name) as fgt:
            gt_line = fgt["lpos"][:, :, :2]

        line, score = line_parsing_from_npz(
            pred_name,
            delta=C.model.delta, nlines=C.model.nlines,
            s_nms=C.model.s_nms, resolution=C.model.resolution
        )
        line = line * (128 / C.model.resolution)

        n_gt += len(gt_line)
        indices = score >= confidence_threshold
        score = score[indices]
        line = line[indices]
        #print(len(line),len(gt_line))
        n_pt += len(line)
        tp, fp, hit = FClip.metric.msTPFP_hit(line, gt_line, threshold)
        tps.append(tp)
        fps.append(fp)
        scores.append(score)

        # Calculating F1score
        indices = np.argsort(score)[::-1]
        cumsum_tp = np.cumsum(np.array(tp)[indices]) / len(gt_line)
        cumsum_fp = np.cumsum(np.array(fp)[indices]) / len(gt_line)
        p = (cumsum_tp[-1]) / np.maximum(cumsum_tp[-1] + cumsum_fp[-1], 1e-9) if len(cumsum_tp) > 0 else 0
        r = cumsum_tp[-1] if len(cumsum_tp) > 0 else 0
        f1score = (2 * 100 * p * r) / (p + r + 1e-9)
        f1_score.append(f1score)
        precision.append(p * 100)
        recall.append(r * 100)
        csv_file.write(f"{p}, {r}, {f1score}\n")
    #print(tps)
    tps = np.concatenate(tps)
    fps = np.concatenate(fps)

    scores = np.concatenate(scores)
    index = np.argsort(-scores)
    lcnn_tp = np.cumsum(tps[index]) / n_gt
    lcnn_fp = np.cumsum(fps[index]) / n_gt
    print(path[-20:-5],FClip.metric.ap(lcnn_tp, lcnn_fp) * 100,np.mean(precision), np.mean(recall), np.mean(f1_score))
    return FClip.metric.ap(lcnn_tp, lcnn_fp) * 100, np.mean(precision), np.mean(recall), np.mean(f1_score)


def batch_sAP_s1(paths, GT, dataname):
    gt = GT

    def work(path):
        #print(f"Working on {path}")
        #try:
        fp = os.path.join(path, "results.csv")
        return [line_center_score(f"{path}/*.npz", gt, t, csv_file_path=fp) for t in [10]]
        #except:
        #   print(f"Issue with {path}")
        #  return [0, 0, 0]

    dirs = sorted(sum([glob.glob(p) for p in paths], []))
    results = FClip.utils.parmap(work, dirs, 8)
    outdir = os.path.dirname(os.path.dirname(args['<path>'][0]))
    print("outdir: ", outdir)
    with open(f"{outdir}/sAP_{dataname}.csv", "a") as fout:
        print(f"nlines: {C.model.nlines}", file=fout)
        print(f"s_nms: {C.model.s_nms}", file=fout)
        for d, msAP in zip(dirs, results):
            total_epochs = int(int(d[-9:])/17796)
            print(f"{d[-13:]}, {total_epochs} epochs : {' '.join(map(lambda x: '%2.1f'%x, msAP[0]))}", file=fout)


def sAP_s1(path, GT):
    sAP = [10]
    print(f"Working on {path}")
    print("sAP: ", sAP)
    return [line_center_score(f"{path}/*.npz", GT, t) for t in sAP]


if __name__ == "__main__":

    args = docopt(__doc__)

    config_file = args["--config"]
    if config_file == "config":
        config_file = args["<path>"][-1]
        config_file = os.path.dirname(config_file)
        config_file = os.path.dirname(config_file)
        config_file = os.path.join(config_file, "config.yaml")
    print(f"load config file {config_file}")
    C.update(C.from_yaml(filename=config_file))

    GT_york = f"/home/dxl/Data/york/valid/*_label.npz"
    GT_huang = f"/home/dxl/Data/wireframe/valid/*_label.npz"
    GT_roofLine = f"../../slakshmi/F-Clip/data/test_gt/*_label.npz"

    print(args["--mode"])
    if args["--mode"] == "shanghaiTech":
        GT = GT_huang
        idx = int(len(args["<path>"]) / 2)

        if idx <= 32:
            batch_sAP_s1(args["<path>"], GT, args["--mode"])
        else:
            batch_sAP_s1(args["<path>"][-idx:], GT, args["--mode"])

        # ------------------------
        C.model.nlines = 1000
        C.model.s_nms = 2
        batch_sAP_s1(args["<path>"][-8:], GT, args["--mode"] + "_nline1k_snms2")
    elif args["--mode"] == "york":
        GT = GT_york
        batch_sAP_s1(args["<path>"], GT, args["--mode"])

        # -------------------------
        C.model.nlines = 1000
        C.model.s_nms = 2
        batch_sAP_s1(args["<path>"], GT, args["--mode"] + "_nline1k_snms2")
    elif args["--mode"] == "roofLine":
        GT = GT_roofLine
        batch_sAP_s1(args["<path>"], GT, args["--mode"])

        C.model.nlines = 1000
        C.model.s_nms = 2
        # batch_sAP_s1(args["<path>"], GT, args["--mode"] + "_nline1k_snms2")
    else:
        print(args["--mode"])
        raise ValueError("no such dataset")


