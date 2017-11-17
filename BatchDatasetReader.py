"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import tensorflow as tf
import numpy as np
import scipy.misc as misc
from PIL import Image
import os, sys


class BatchDatasetReader:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}, data_dir=""):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.data_dir = data_dir
        self.image_options = image_options
        self.images_filename = "JPEGImages_tranformed"
        self.annotations_filename = "SegmentationClass_tranformed"
        self._read_images()
        

    def _read_images(self):
        self.__channels = True
        images_filepath = os.path.join(self.data_dir, self.images_filename)
        annotations_filepath = os.path.join(self.data_dir, self.annotations_filename)
        self.preprocess(images_filepath, "images")
        self.__channels = False
        self.preprocess(annotations_filepath, "annotations")
        print(self.images.shape)
        print(self.annotations.shape)

    def _transform(self, filename):
        image = np.array(Image.open(filename))
        img_h, img_w = image.shape[0], image.shape[1]

        # ensure that the pictures and the annotations have the same dimensions
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])
            # 转置维度
            image = image.transpose([1, 2, 0])

        # fill the image to 500x500
        if img_h < 500 or img_w < 500:
            img_fill = np.zeros([500, 500, 3], 'int32') if len(image.shape) == 3 else np.zeros([500, 500], 'int32')
            
            img_h_fill = (500 - img_h) // 2
            img_w_fill = (500 - img_w) // 2
            for idx_i, i in enumerate(image):
                for idx_j, j in enumerate(i):
                    img_fill[idx_i + img_h_fill][idx_j + img_w_fill] = j
            image = img_fill
            del img_fill

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='bilinear')
        else:
            resize_image = image
        
        if len(image.shape) == 2:
            a = tf.constant(250, shape=[500,500], dtype=tf.int32)
            less_than_255 = tf.cast(tf.less(image, a), dtype=tf.int32)
            resize_image = less_than_255 * image
            sess = tf.Session()
            resize_image = resize_image.eval(session=sess)
            

        return np.array(resize_image)

    def preprocess(self, filepath, filetype):
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            if filetype == "images":
                self.images = np.array([self._transform(filename) for filename in self.files[filetype]])
            else:
                self.annotations = np.array([self._transform(filename) for filename in self.files[filetype]])
            self.save_records(filetype)
        else:
            origin = "JPEGImages" if filetype == "images" else "SegmentationClass"
            transformed = self.images_filename if filetype == "images" else self.annotations_filename
            if filetype == "images":
                self.images = np.array([np.array(Image.open(filename.replace(origin, transformed))) for filename in self.files[filetype]])
            else:
                self.annotations = np.array([np.array(Image.open(filename.replace(origin, transformed))) for filename in self.files[filetype]])
            
    def save_records(self, filetype):
        i = 0
        origin = "JPEGImages" if filetype == "images" else "SegmentationClass"
        transformed = self.images_filename if filetype == "images" else self.annotations_filename
        for filename in self.files[filetype]:
            if filetype == "images":
                new_img = Image.fromarray(self.images[i])
                new_img.save(filename.replace(origin, transformed))
            else:
                # 获取颜色表
                pal = (Image.open(filename)).getpalette()
                print(self.annotations.shape)
                new_img = Image.fromarray(self.annotations[i], mode="P")
                # 设置颜色表
                new_img.putpalette(pal)
                new_img.save(filename.replace(origin, transformed))
            i += 1

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]