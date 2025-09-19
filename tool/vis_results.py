import glob
import numpy as np
import os
from visual_utils import open3d_vis_utils as V


def load_pred_file(pred_file):
    data = np.loadtxt(pred_file)
    if data.ndim == 1:  # 只有一行时，保证 shape 正确
        data = data[np.newaxis, :]

    boxes = data[:, 0:7]
    labels = data[:, 7].astype(int) + 1
    scores = data[:, 8]
    return boxes, labels, scores


data_path = "/media/taole/mydisk/DL_PROJECT/CUDA-PointPillars-detailed/data"
ptc_files = glob.glob(os.path.join(data_path, "*.bin"))

for ptc_file in ptc_files:
    pred_file = os.path.splitext(ptc_file)[0] + ".txt" # splitext拆分为文件名和扩展名
    ptcs = np.fromfile(ptc_file, dtype=np.float32).reshape(-1, 4)
    pred_boxes, pred_labels, pred_scores = load_pred_file(pred_file)
    V.draw_scenes(points=ptcs, ref_boxes=pred_boxes, ref_scores=pred_scores, ref_labels=pred_labels)

