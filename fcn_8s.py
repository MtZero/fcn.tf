from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from vgg16_model import vgg16 as vgg16
import TensorflowUtils as utils
import read_voa_data as scene_parsing
import datetime
import BatchDatasetReader as dataset
from six.moves import xrange
from function import accuracy
from PIL import Image

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "10", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "dataset/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Momentum Optimizer")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
tf.flags.DEFINE_float("momentum", "0.9", "momentum for Momentum Optimizer")
tf.flags.DEFINE_float("weight_decay", "5e-4", "weight_decay for reg_loss")

MAX_ITERATION = int(1e5 + 1)
IMAGE_SIZE = 500

MODEL_URL = "http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-16.mat"


def inference(image, keep_prob):
    """ fcn_8s """
    # load data
    vgg16_object = vgg16()
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = model_data['layers'][0]

    # preprocess
    processed_image = utils.process_image(image, mean_pixel)

    # deconvolution
    with tf.variable_scope("inference"):
        image_net = vgg16_object._vgg16_modified(processed_image, weights)

        pool4, pool3 = image_net["pool4"], image_net["pool3"]

        conv_final_layer = image_net["conv5_3"]

        """ pool5 """
        pool5 = vgg16_object._max_pool(conv_final_layer, 2, 2, 'pool5')

        """ fc6 """
        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        fc6 = vgg16_object._conv(pool5, W6, b6, 'fc6')
        relu6 = vgg16_object._relu(fc6, 'relu6')
        fc6 = tf.nn.dropout(relu6, keep_prob)

        """ fc7 """
        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        fc7 = vgg16_object._conv(fc6, W7, b7, 'fc7')
        relu7 = vgg16_object._relu(fc7, 'relu7')
        fc7 = tf.nn.dropout(relu7, keep_prob)

        """ fc8 """
        W8 = utils.weight_variable([1, 1, 4096, 21], name="W8")
        b8 = utils.bias_variable([21], name="b8")
        fc8 = vgg16_object._conv(fc7, W8, b8, 'fc8')

        # upsample and add with pool4(pool4 should pass through a score layer)
        score_w1 = utils.weight_variable([1,1,512,21], name="score_w1")
        score_b1 = utils.weight_variable([21], name="score_b1")
        score1 = vgg16_object._conv(pool4, score_w1, score_b1, 'score1')
        W_t1 = tf.Variable(initial_value=bilinear_init(), name="W_t1")
        b_t1 = tf.Variable(initial_value=tf.constant(0.0, shape=[21]), name="b_t1")
        conv_t1 = vgg16_object.conv2d_transpose_strided(fc8, W_t1, b_t1, output_shape=tf.shape(score1))
        fuse_1 = tf.add(conv_t1, score1, name="fuse_1")

        # upsample and add with pool3(pool3 should pass through a score layer)
        score_w2 = utils.weight_variable([1,1,256,21], name="score_w2")
        score_b2 = utils.weight_variable([21], name="score_b2")
        score2 = vgg16_object._conv(pool3, score_w2, score_b2, 'score2')
        W_t2 = tf.Variable(initial_value=bilinear_init(), name="W_t2")
        b_t2 = tf.Variable(initial_value=tf.constant(0.0, shape=[21]), name="b_t2")
        conv_t2 = vgg16_object.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(score2))
        fuse_2 = tf.add(conv_t2, score2, name="fuse_2")

        shape = tf.shape(processed_image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], 21])
        W_t3 = tf.Variable(initial_value=bilinear_init(scale=8), name="W_t3")
        b_t3 = tf.Variable(initial_value=tf.constant(0.0, shape=[21]), name="b_t3")
        conv_t3 = vgg16_object.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        prediction = tf.argmax(conv_t3, dimension=3, name="prediction")
        return prediction, conv_t3, pool4

#Create bilinear weights in numpy array
def bilinear_init(scale=2, num_classes=21):
    filter_size = (2 * scale - scale % 2)
    bilinear_kernel = np.zeros([filter_size, filter_size], dtype=np.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            bilinear_kernel[x,y] = (1 - abs(x - center) / scale_factor) * \
                                   (1 - abs(y - center) / scale_factor)
    weights = np.zeros((filter_size, filter_size, num_classes, num_classes))
    for i in range(num_classes):
        weights[:, :, i, i] = bilinear_kernel

    #assign numpy array to tensor
    bilinear_initial = tf.convert_to_tensor(weights, dtype=tf.float32)
    return bilinear_initial


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
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE], name="annotation")

    # TODO
    pred_annotation, logits, c_l_l = inference(image, keep_probability)
    
    tf.summary.image("input_image", image, max_outputs=20)
    # tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=20)
    # tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=20)
    annotation_onehot = tf.one_hot(annotation, 21, 1.0, 0.0, -1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = annotation_onehot, logits = logits))
    # loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.clip_by_value(tf.cast(logits, dtype=tf.float32), 1e-10, 1), labels=tf.cast(annotation, dtype=tf.int32))))
    tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
        # reg_loss = tf.add_n(tf.get_collection("reg_loss"))
    reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    loss += FLAGS.weight_decay * reg_loss
    

    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': False, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatasetReader(train_records, image_options, FLAGS.data_dir)
        validation_dataset_reader = dataset.BatchDatasetReader(valid_records, image_options, FLAGS.data_dir)
    
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.7
    sess = tf.Session(config=tf_config)
    
    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.initialize_all_variables())
    # print(sess.run(tf.trainable_variables()))
    
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    
    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.read_next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)
            print(sess.run(c_l_l, feed_dict=feed_dict))
            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.read_next_batch(FLAGS.batch_size)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                keep_probability: 1.0})
                valid_accu = accuracy.batch_calc_accuracy(pred, valid_annotations)
                print("%s ---> Validation_accu: %g" % (datetime.datetime.now(), valid_accu))
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        # pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
        #                                             keep_probability: 1.0})
        # pred = np.squeeze(pred, axis=3)

        # save the prediction
        pass
        # for itr in range(FLAGS.batch_size):
        #     # save gt
        #     pal = (Image.open('/home/jingyang/Desktop/fcn/fcn_test/dataset/SegmentationClass_tranformed/2007_000032.png')).getpalette()
        #     new_img = Image.fromarray(valid_annotations[itr], mode="P")
        #     new_img.putpalette(pal)
        #     path = "logs/gt_"+ str(5 + itr)+".png"
        #     new_img.save(path)
        #     # save inp
        #     utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5 + itr))
        #     #utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5 + itr))
        #     # save pred
        #     new_img = Image.fromarray(pred[itr], mode="P")
        #     new_img.putpalette(pal)
        #     path = "logs/pred_"+ str(5 + itr)+".png"
        #     new_img.save(path)
        #     # utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5 + itr))
        #     print("Saved image: %d" % itr)


if __name__ == "__main__":
    tf.app.run()