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
        self.train_dataset_size = len(self.files['images'])
        self.perm = np.arange(self.train_dataset_size)
        self.data_dir = data_dir
        self.image_options = image_options
        self.count = 0
        exp_annotation = 'dataset/SegmentationClass/2007_000032.png'
        # 获取颜色表
        self.pal = (Image.open(exp_annotation)).getpalette()
        
    def read_next_batch(self, batch_size):
        self._clear_memory()
        # make file
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.train_dataset_size:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            np.random.shuffle(self.perm)
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset
        self._read_images(self.perm[start:end])
        print(self.images.shape)
        return self.images, self.annotations

    def _read_images(self, read_batch_list):
        images_filepath = os.path.join(self.data_dir, "JPEGImages_tranformed")
        annotations_filepath = os.path.join(self.data_dir, "SegmentationClass_tranformed")
        self._preprocess(images_filepath, 'images', read_batch_list)
        self._preprocess(annotations_filepath, 'annotations', read_batch_list)
        

    def _preprocess(self, filepath, filetype, read_batch_list):
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        origin = "JPEGImages" if filetype == "images" else "SegmentationClass"
        transformed = "JPEGImages_tranformed" if filetype == "images" else "SegmentationClass_tranformed"
        print(len(self.files[filetype]))
        img_list = np.array(self.files[filetype])[read_batch_list]
        for filename in img_list:
            img_transformed = ''
            if not os.path.exists(filename.replace(origin, transformed)):
                img_transformed = self._transform(filename)
                # save to memory
                if filetype == "images":
                    self.images = np.append(self.images, img_transformed, axis=0) if self.images != [] else img_transformed
                    print(self.images.shape)
                elif filetype == "annotations":
                    self.annotations = np.append(self.annotations, img_transformed, axis=0) if self.annotations != [] else img_transformed
                # save to local
                self._save_img(filename.replace(origin, transformed), filetype)
            else:
                img_transformed = np.array(Image.open(filename.replace(origin, transformed)))
                img_transformed = np.array([img_transformed], dtype=np.uint8)
                if filetype == "images":
                    self.images = np.append(self.images, img_transformed, axis=0) if self.images != [] else img_transformed
                elif filetype == "annotations":
                    self.annotations = np.append(self.annotations, img_transformed, axis=0) if self.annotations != [] else img_transformed
            
        

    def _transform(self, filename):
        image = np.array(Image.open(filename))
        image_name = filename.split('/')[2]
        img_h, img_w = image.shape[0], image.shape[1]

        # scale to 500xN
        if img_h != 500 and img_w != 500:
            img_h = 500 if img_h > img_w else int(500/img_w*img_h)
            img_w = 500 if img_h <= img_w else int(500/img_h*img_w)
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [img_h, img_w], interp='bilinear')
        # fill the image to 500x500
        resize_image = []
        if img_h < 500 or img_w < 500:
            img_fill = np.zeros([500, 500, 3], dtype=np.uint8) if len(image.shape) == 3 else np.ones([500, 500], dtype=np.uint8)*255
            img_h_fill = (500 - img_h) // 2
            img_w_fill = (500 - img_w) // 2
            for idx_i, i in enumerate(image):
                for idx_j, j in enumerate(i):
                    img_fill[idx_i + img_h_fill][idx_j + img_w_fill] = j
            resize_image = img_fill
            del img_fill
        else:
            resize_image = image
            
        if len(image.shape) == 2:
            resize_image[resize_image == 255] = 21
        else:
            self.count += 1
            print('complete', self.count, ":", image_name)
        return np.array(np.array([resize_image]), dtype=np.uint8)
        
    def _save_img(self, filename, filetype):
        if filetype == "images":
            new_img = Image.fromarray(self.images[-1])
        else:
            new_img = Image.fromarray(self.annotations[-1], mode="P")
            # 设置颜色表
            new_img.putpalette(self.pal)
        new_img.save(filename)

    def get_random_batch(self, batch_size):
        batch_list = np.random.randint(0, self.train_dataset_size, size=[batch_size]).tolist()
        self._read_images(batch_list)
        return self.images, self.annotations

    def _clear_memory(self):
        if self.images != []:
            del self.images
        if self.annotations != []:
            del self.annotations
        self.images = []
        self.annotations = []
