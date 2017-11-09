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
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Momentum Optimizer")tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
tf.flags.DEFINE_float("momentum", "0.9", "momentum for Momentum Optimizer")

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
    optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)
    
def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    tf.image_summary("input_image", image, max_images=20)
    tf.image_summary("ground_truth", tf.cast(annotation, tf.uint8), max_images=20)
    tf.image_summary("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_images=20)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
                                                                          tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
    tf.scalar_summary("entropy", loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.merge_all_summaries()

    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatasetReader(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatasetReader(valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.initialize_all_variables())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr)