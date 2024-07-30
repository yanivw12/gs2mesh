import json
import numpy as np


def load_extr(path):
    with open(path, "r") as f:
        cam = json.load(f)
    T_wc = np.asarray(cam['cameraPoseARFrame']).reshape(4, 4)
    T_align = np.eye(4)
    T_align[1, 1] = -1
    T_align[2, 2] = -1
    T_wc = T_wc @ T_align
    return T_wc

def load_intr(path):
    with open(path, "r") as f:
        cam = json.load(f)
    intr_mat = np.asarray(cam['intrinsics']).reshape(3, 3)
    return intr_mat


def load_alignment(path):
    with open(path, "r") as f:
        cam = json.load(f)
    align_mat = np.asarray(cam['alignment'][:-1]).reshape(4, 4)
    scale = float(cam['alignment'][-1])
    return align_mat, scale


