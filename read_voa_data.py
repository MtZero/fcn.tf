import numpy as np
import os
import glob
import TensorflowUtils as utils
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile


def read_dataset(data_dir):
    pickle_filename = "VOA"
