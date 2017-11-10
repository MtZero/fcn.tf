"""vgg16 model.

Related papers:
https://arxiv.org/pdf/1409.1556.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import TensorflowUtils as utils

class vgg16(object):
    """vgg16 model."""

    # def __init__(self, is_training, data_format, batch_norm_decay, batch_norm_epsilon):
    def __init__(self):
        """vgg16 constructor.

        Args:
          is_training: if build training or inference model.
          data_format: the data_format used during computation.
                       one of 'channels_first' or 'channels_last'.
        """
        # self._batch_norm_decay = batch_norm_decay
        # self._batch_norm_epsilon = batch_norm_epsilon
        # self._is_training = is_training
        # assert data_format in ('channels_first', 'channels_last')
        # self._data_format = data_format
        self._kernel_size = 3
        self._stride = 1
        self.pool_size = 2
        self.pool_stride = 2

    def forward_pass(self, x):
        raise NotImplementedError(
            'forward_pass() is implemented in ResNet sub classes')

    def _vgg16_modified(self, x, weights):
        with tf.name_scope('vgg16') as name_scope:
            orig_x = x

            layers = (
                'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

                'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

                'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
                'relu3_3', 'pool3',

                'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
                'relu4_3', 'pool4',

                'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
                'relu5_3'
            )
            # x = self._batch_norm(x)
            # x = self._relu(x)
            net = {}
            current = x
            for i, name in enumerate(layers):
                kind = name[:4]
                if kind == 'conv':
                    kernels, bias = weights[i][0][0][0][0]
                    # matconvnet: weights are [width, height, in_channels, out_channels]
                    # tensorflow: weights are [height, width, in_channels, out_channels]
                    kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
                    bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                    current = self._conv(current, kernels, bias, name)
                elif kind == 'relu':
                    current = self._relu(current, name=name)
                elif kind == 'pool':
                    current = self._max_pool(current, self.pool_size, self.pool_stride, name)
                net[name] = current

            return net
            

    def _conv(self, x, W, bias, name):
        """Convolution."""
        with tf.variable_scope(name) as scope:
            conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
            return tf.nn.bias_add(conv, bias)
            # bn = self._batch_norm(x)

        
    def _batch_norm(self, x):
        if self._data_format == 'channels_first':
            data_format = 'NCHW'
        else:
            data_format = 'NHWC'
        return tf.contrib.layers.batch_norm(
            x,
            decay=self._batch_norm_decay,
            center=True,
            scale=True,
            epsilon=self._batch_norm_epsilon,
            is_training=self._is_training,
            fused=True,
            data_format=data_format)

    def _relu(self, x, name):
        return tf.nn.relu(x, name=name)

    def _fully_connected(self, x, out_dim):
        with tf.name_scope('fully_connected') as name_scope:
            x = tf.layers.dense(x, out_dim)

        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def _avg_pool(self, x, pool_size, stride):
        with tf.name_scope('avg_pool') as name_scope:
            x = tf.layers.average_pooling2d(
                x, pool_size, stride, 'SAME', data_format=self._data_format)

        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def _max_pool(self, x, pool_size, stride, name):
        with tf.name_scope(name) as name_scope:
            x = tf.layers.max_pooling2d(x, pool_size, stride, 'SAME')

        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def conv2d_transpose_strided(self, x, W, b, output_shape=None, stride=2):
        if output_shape is None:
            output_shape = x.get_shape().as_list()
            output_shape[1] *= 2
            output_shape[2] *= 2
            output_shape[3] = W.get_shape().as_list()[2]
        conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
        return tf.nn.bias_add(conv, b)
