import tensorflow as tf
import numpy as np
from PIL import Image


def calc_accuracy(a, b, split_class=21):
    """
    Calculate the accuracy of one prediction.
    :param a: the prediction in numpy.ndarray, should be in 2D
    :param b: the ground truth with the same shape of prediction.
    :param split_class: the number of classes
    :return: accuracy, using IoU
    """
    if not (isinstance(a, np.ndarray) & isinstance(b, np.ndarray)):
        raise TypeError("The type of args should be numpy.ndarray!")
    if a.shape != b.shape:
        raise AttributeError("The inputs should be in the same shape!")
    if len(a.shape) != 2:
        raise AttributeError("The inputs should be 2-dimension!")
    ones = np.ones(a.shape, dtype="int32")
    tmp = split_class * ones
    edge = b != tmp
    tmp = np.zeros(a.shape, dtype="int32")
    cnt = 0.
    accuracy = 0.
    for i in range(split_class):
        a_class = ((a == tmp) & edge)
        b_class = (b == tmp)
        union = np.sum(a_class | b_class)
        if union != 0:
            cnt += 1
            inter = np.sum(a_class & b_class)
            accuracy += inter / union
        tmp = tmp + ones
    return accuracy / cnt


if __name__ == "__main__":
    a = np.array(Image.open("../dataset/2007_000032.png"), dtype=np.int8)
    a[130:140, :] = 0
    b = np.array(Image.open("../dataset/2007_000032.png"), dtype=np.int8)
    b[b == 255] = 21
    print(calc_accuracy(a, b))
