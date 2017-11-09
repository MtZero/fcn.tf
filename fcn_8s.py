from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import vgg16_model as vgg16
import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatasetReader as dataset

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "20", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")


def inference(self, image, keep_prob):
    """ fcn_8s """
    # load data
    pass

    # preprocess
    pass

    # deconvolution
    with tf.variable_scope("inference"):
        pool3, pool4, fc8 = vgg16._vgg16_modified(image_input, 1, keep_prob, True)
        # upsample and add with pool4
        deconv_shape1 = pool4.get_shape()
        W_t1 = tf.Variable(initializer=tf.truncated_normal([4, 4, deconv_shape1[3].value, 21], 0.02), name="W_t1")
        b_t1 = tf.Variable(initializer=tf.constant(0.0, shape=[deconv_shape1[3].value]), name="b_t1")
        conv_t1 = vgg16.conv2d_transpose_strided(fc8, W_t1, b_t1, output_shape=tf.shape(pool4))
        fuse_1 = tf.add(conv_t1, pool4, name="fuse_1")

        # upsample and add with pool5
        deconv_shape2 = pool3.get_shape()
        W_t2 = tf.Variable(initializer=tf.truncated_normal([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], 0.02), name="W_t2")
        b_t2 = tf.Variable(initializer=tf.constant(0.0, shape=[deconv_shape2[3].value]), name="b_t2")
        conv_t2 = vgg16.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(pool3))
        fuse_2 = tf.add(conv_t2, pool3, name="fuse_2")

        shape = tf.shape(image_input)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], 21])
        W_t3 = tf.Variable(initializer=tf.truncated_normal([16, 16, 21, deconv_shape2[3].value], 0.02), name="W_t3")
        b_t3 = tf.Variable(initializer=tf.constant(0.0, shape=[21]), name="b_t3")
        conv_t3 = vgg16.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        prediction = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(prediction, dim=3), conv_t3

def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)
    
def main(argv=None):
    pass