import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.io
import os
from data import utils as CTRUtil


DATA_DIR = "data/"
JSRT_DIR = DATA_DIR + "jsrt/"
WING_DIR = DATA_DIR + "wingspan/"
JSRT_FNAMES = os.listdir(JSRT_DIR + "png")
WING_FNAMES = os.listdir(WING_DIR + "png")


def get_input_image_example(base_dir, fname):
    """
    Parameter
        base_dir = JSRT_DIR or WING_DIR,
        fname = one of JSRT_FNAMES or WING_FNAMES
    Returns numpy array representing image with 1 BW channel.
    Range of each value is [0.0, 1.0]
    Shape: (height, width, 1 channel)
    """
    img = skimage.io.imread(base_dir + "png/" + fname)
    norm = img / 255.0
    return np.expand_dims(norm, -1)


def get_input_image_examples(base_dir, fnames):
    """
    Accumulates get_input_image_example.
    Shape: (#examples, height, width, 1 channel)
    """
    return np.array([
        get_input_image_example(base_dir, fname)
        for fname in fnames
    ])


def get_ground_truth_segments(base_dir, fname):
    """
    Parameter
        base_dir = JSRT_DIR or WING_DIR,
        fname = one of JSRT_FNAMES or WING_FNAMES
    Returns numpy array representing image with 4 channels (one-hot):
        0: none
        1: Left lung mask
        2: Right lung mask
        3: Heart mask
    Shape: (height, width, 4 channels)
    """
    left = skimage.io.imread(base_dir + "mask/left_lung/" + fname)
    right = skimage.io.imread(base_dir + "mask/right_lung/" + fname)
    heart = skimage.io.imread(base_dir + "mask/heart/" + fname)

    for mask in (left, right, heart):
        mask[mask > 0] = 1.0

    non = np.ones(left.shape) - left - right - heart
    non[non < 0] = 0.0
    non[non > 0] = 1.0

    concat = np.stack((non, left, right, heart), axis=2)
    return concat


def get_ground_truth_segment_set(base_dir, fnames):
    """
    Accumulates get_ground_truth_segments.
    Shape: (#examples, height, width, 4 channels)
    """
    return np.array([
        get_ground_truth_segments(base_dir, fname)
        for fname in fnames
    ])


def get_data(base_dir, fnames):
    """
    Calls get_input_image_examples on the given
        base_dir : ex. JSRT_DIR
        fnames : ex. JSRT_FNAMES
    Returns:
        set of images, 
        set of segments
    """
    X = get_input_image_examples(base_dir, fnames)
    Y = get_ground_truth_segment_set(base_dir, fnames)
    return X, Y


def save_data(includeJSRT, includeWingspan, imgName, segName, relpath):
    assert includeJSRT ^ includeWingspan, "Must include exacly one dataset"

    X, Y = None, None

    if includeJSRT:
        X, Y = get_data(JSRT_DIR, JSRT_FNAMES)
    else:
        X, Y = get_data(WING_DIR, WING_FNAMES)

    print("Saving images of shape {} and segments of shape {} to disk".format(
        X.shape, Y.shape))

    for name, arr in [(imgName, X), (segName, Y)]:
        np.save("{}{}.npy".format(relpath, name), arr)


def save_all_data(imgName="imgs", segName="segs", relpath="./"):
    print("Saving JSRT...")
    save_data(True, False, "JSRT_"+imgName, "JSRT_"+segName, relpath)
    print("Saving Wingspan...")
    save_data(False, True, "Wingspan_"+imgName, "Wingspan_"+segName, relpath)
    print("Done!")


def load_data(imgName, segName, relpath="./"):
    imgs, segs = None, None

    with open('{}{}.npy'.format(relpath, imgName), 'rb') as f:
        imgs = np.load(f)

    with open('{}{}.npy'.format(relpath, segName), 'rb') as f:
        segs = np.load(f)

    return imgs, segs


if __name__ == "__main__":
    if not os.path.exists('prepared_data'):
        os.makedirs('prepared_data')
    save_all_data(relpath="./prepared_data/")
