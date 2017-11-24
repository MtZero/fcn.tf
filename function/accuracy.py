import tensorflow as tf
import numpy as np
from PIL import Image


def calc_accuracy(prediction_in, groundtruth_in, split_class=21):
    """
    Calculate the accuracy of one prediction.
    :param prediction_in: the prediction in numpy.ndarray, should be in 2D
    :param groundtruth_in: the ground truth with the same shape of prediction.
    :param split_class: the number of classes
    :return: accuracy, using IoU
    """
    if not (isinstance(prediction_in, np.ndarray) & isinstance(groundtruth_in, np.ndarray)):
        raise TypeError("The type of inputs should be numpy.ndarray!")
    if prediction_in.shape != groundtruth_in.shape:
        raise AttributeError("The inputs should be in the same shape!")
    if len(prediction_in.shape) != 2:
        raise AttributeError("The inputs should be 2-dimension!")
    ones = np.ones(prediction_in.shape, dtype="int32")
    tmp = split_class * ones
    edge = groundtruth_in != tmp
    tmp = np.zeros(prediction_in.shape, dtype="int32")
    cnt = 0.
    accuracy = 0.
    for i in range(split_class):
        a_class = ((prediction_in == tmp) & edge)
        b_class = (groundtruth_in == tmp)
        union = np.sum(a_class | b_class)
        if union != 0:
            cnt += 1
            inter = np.sum(a_class & b_class)
            accuracy += inter / union
        tmp = tmp + ones
    return accuracy / cnt


def batch_calc_accuracy(prediction_in, groundtruth_in, split_class=21):
    """
    batch calculate accuracy
    :param prediction_in: the prediction in numpy.ndarray, should be in shape[N,x,x]
    :param groundtruth_in: the ground truth with the same shape of prediction
    :param split_class: the number of classes
    :return: batch accuracy
    """
    if not (isinstance(prediction_in, np.ndarray) & isinstance(groundtruth_in, np.ndarray)):
        raise TypeError("The type of inputs should be numpy.ndarray!")
    if prediction_in.shape != groundtruth_in.shape:
        raise AttributeError("The inputs should be in the same shape!")
    if len(prediction_in.shape) != 3:
        raise AttributeError("The inputs should be 3-dimension!")
    accuracy = 0.
    for i in range(prediction_in.shape[0]):
        accuracy += calc_accuracy(prediction_in[i], groundtruth_in[i])
    return accuracy / prediction_in.shape[0]


if __name__ == "__main__":
    a = np.array(Image.open("../dataset/2007_000032.png"), dtype=np.int8)
    a[100:180, :] = 0
    b = np.array(Image.open("../dataset/2007_000032.png"), dtype=np.int8)
    b[b == 255] = 21
    print(batch_calc_accuracy(a.reshape(1, a.shape[0], a.shape[1]), b.reshape(1, b.shape[0], b.shape[1])))
