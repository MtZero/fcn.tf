"""vgg16 model.

Related papers:
https://arxiv.org/pdf/1409.1556.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class vgg16(object):
    """vgg16 model."""

    def __init__(self, is_training, data_format, batch_norm_decay, batch_norm_epsilon):
        """vgg16 constructor.

        Args:
          is_training: if build training or inference model.
          data_format: the data_format used during computation.
                       one of 'channels_first' or 'channels_last'.
        """
        self._batch_norm_decay = batch_norm_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        self._is_training = is_training
        assert data_format in ('channels_first', 'channels_last')
        self._data_format = data_format
        self._kernel_size = 3
        self._stride = 1
        self.pool_size = 3
        self.pool_stride = 2

    def forward_pass(self, x):
        raise NotImplementedError(
            'forward_pass() is implemented in ResNet sub classes')

    def _vgg16_modified(self, x, stride, keep_prob, train=False):
        with tf.name_scope('vgg16') as name_scope:
            orig_x = x

            # x = self._batch_norm(x)
            # x = self._relu(x)

            """ conv_1 """
            self.conv1_1 = self._conv(x, self._kernel_size, 64, self._stride, 'conv1_1')
            self.conv1_2 = self._conv(self.conv1_1, self._kernel_size, 64, self._stride, 'conv1_2')
            self.pool1 = self._max_pool(self.conv1_2, self.pool_size, self.pool_stride, 'max_pool1')

            """ conv_2 """
            self.conv2_1 = self._conv(self.pool1, self._kernel_size, 128, self._stride, 'conv2_1')
            self.conv2_2 = self._conv(self.conv2_1, self._kernel_size, 128, self._stride, 'conv2_2')
            self.pool2 = self._max_pool(self.conv2_2, self.pool_size, self.pool_stride, 'max_pool2')

            """ conv_3 """
            self.conv3_1 = self._conv(self.pool2, self._kernel_size, 256, self._stride, 'conv3_1')
            self.conv3_2 = self._conv(self.conv3_1, self._kernel_size, 256, self._stride, 'conv3_2')
            self.conv3_3 = self._conv(self.conv3_2, self._kernel_size, 256, self._stride, 'conv3_3')
            self.pool3 = self._max_pool(self.conv3_3, self.pool_size, self.pool_stride, 'max_pool3')

            """ conv_4 """
            self.conv4_1 = self._conv(self.pool3, self._kernel_size, 512, self._stride, 'conv4_1')
            self.conv4_2 = self._conv(self.conv4_1, self._kernel_size, 512, self._stride, 'conv4_2')
            self.conv4_3 = self._conv(self.conv4_2, self._kernel_size, 512, self._stride, 'conv4_3')
            self.pool4 = self._max_pool(self.conv4_3, self.pool_size, self.pool_stride, 'max_pool4')

            """ conv_5 """
            self.conv5_1 = self._conv(self.pool4, self._kernel_size, 512, self._stride, 'conv5_1')
            self.conv5_2 = self._conv(self.conv5_1, self._kernel_size, 512, self._stride, 'conv5_2')
            self.conv5_3 = self._conv(self.conv5_2, self._kernel_size, 512, self._stride, 'conv5_3')
            self.pool5 = self._max_pool(self.conv5_3, self.pool_size, self.pool_stride, 'max_pool5')

            """ fc6 """
            self.fc6 = self._conv(self.pool5, 7, 4096, self._stride, 'fc6')
            if train:
                self.fc6 = tf.nn.dropout(self.fc6, keep_prob)

            """ fc7 """
            self.fc7 = self._conv(self.fc6, 1, 4096, self._stride, 'fc7')
            if train:
                self.fc7 = tf.nn.dropout(self.fc7, keep_prob)

            """ fc8 """
            self.fc8 = self._conv(self.fc8, 1, 21, self._stride, 'fc8')

            return self.pool3, self.pool4, self.fc8

    def _conv(self, x, kernel_size, filters, strides, name):
        """Convolution."""
        with tf.variable_scope(name) as scope:
            padding = 'SAME'
            pad = kernel_size - 1
            pad_beg = pad // 2
            pad_end = pad - pad_beg
            if self._data_format == 'channels_first':
                x = tf.pad(x, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
            else:
                x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
            x = tf.layers.conv2d(
                inputs=x,
                kernel_size=kernel_size,
                filters=filters,
                strides=strides,
                padding=padding,
                use_bias=False,
                data_format=self._data_format)

            # bn = self._batch_norm(x)
            relu = self._relu(x)
            # Add summary to Tensorboard
            _activation_summary(relu)
            return relu

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

    def _relu(self, x):
        return tf.nn.relu(x)

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
            x = tf.layers.max_pooling2d(
                x, pool_size, stride, 'SAME', data_format=self._data_format)

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
