from multiprocessing.spawn import prepare
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.io
import os
from data import utils as CTRUtil

from tqdm import tqdm
import matplotlib.pyplot as plt


# Convert one-hot to 0,1,2,3
def prepare_seg(one_hot):
    return np.argmax(one_hot, axis=2)


def prepare_segs(one_hot_arr):
    return np.argmax(one_hot_arr, axis=3)


def show_annotation(img, segmentation):
    seg = prepare_seg(segmentation)
    annotated = CTRUtil.add_seg(img, seg)
    skimage.io.imshow(annotated)


def show_annotation_pred(img, x):
    seg = np.argsort(np.max(x, axis=2))[-2]
    annotated = CTRUtil.add_seg(img, seg)
    skimage.io.imshow(annotated)

# def visualize_predictions(segmentor, )
