# import numpy as np
import os
# import glob
# import TensorflowUtils as utils
# from six.moves import cPickle as pickle
# from tensorflow.python.platform import gfile


def read_dataset(data_dir):
    # pickle_filename = "VOC.pickle"
    # pickle_filepath = os.path.join(data_dir, pickle_filename)
    # if not os.path.exists(pickle_filepath):
    image_set_dir = os.path.join(data_dir, "ImageSets", "Segmentation")
    train_set_index_file = open(os.path.join(image_set_dir, "train.txt"), "r")
    valid_set_index_file = open(os.path.join(image_set_dir, "val.txt"), "r")
    train_set_index = [x.strip(" \n") for x in train_set_index_file.readlines()]
    valid_set_index = [x.strip(" \n") for x in valid_set_index_file.readlines()]
    image = {"training": {"images": [], "annotations": []}, "validation": {"images": [], "annotations": []}}
    images_path = os.path.join(data_dir, "JPEGImages")
    annotations_path = os.path.join(data_dir, "SegmentationClass")
    for line in train_set_index:
        image["training"]["images"].append(os.path.join(images_path, line + ".jpg"))
        image["training"]["annotations"].append(os.path.join(annotations_path, line + ".png"))
    for line in valid_set_index:
        image["validation"]["images"].append(os.path.join(images_path, line + ".jpg"))
        image["validation"]["annotations"].append(os.path.join(annotations_path, line + ".png"))
    return image
