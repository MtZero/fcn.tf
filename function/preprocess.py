import numpy as np
from PIL import Image

file_path_prefix = "../dataset/"
jpeg_path_prefix = "../dataset/JPEGImages/"
png_path_prefix = "../dataset/SegmentationClass/"
# 一开始的时候拼错了= =现在干脆不改
jpeg_trans_path_prefix = "../dataset/JPEGImages_tranformed/"
png_trans_path_prefix = "../dataset/SegmentationClass_tranformed/"

trainval_file = open(file_path_prefix + "ImageSets/Segmentation/trainval.txt", "r")
trainval_path_list = []

if __name__ == "__main__":
    for line in trainval_file:
        line = line.strip(" \n")
        trainval_path_list.append([line + ".jpg", line + ".png"])
    pal = Image.open(png_path_prefix + trainval_path_list[0][1]).getpalette()
    for idx, line in enumerate(trainval_path_list):
        print("processing #%d" % idx)
        # process JPEG images
        image = np.array(Image.open(jpeg_path_prefix + line[0]))
        image_height, image_width = image.shape[0], image.shape[1]
        # fill to 500 x 500
        image_fill = np.zeros([500, 500, 3], dtype=np.uint8)
        image_height_fill = (500 - image_height) // 2
        image_width_fill = (500 - image_width) // 2
        image_fill[image_height_fill: image_height + image_height_fill,
                   image_width_fill: image_width + image_width_fill] = image
        save_image = Image.fromarray(image_fill)
        save_image.save(jpeg_trans_path_prefix + line[0])
        # process PNG annotations
        annotation = np.array(Image.open(png_path_prefix + line[1]))
        annotation[annotation == 255] = 21
        annotation_height, annotation_width = annotation.shape[0], annotation.shape[1]
        # fill to 500 x 500
        annotation_fill = 21 * np.ones([500, 500], dtype=np.uint8)
        annotation_height_fill = (500 - annotation_height) // 2
        annotation_width_fill = (500 - annotation_width) // 2
        annotation_fill[annotation_height_fill: annotation_height + annotation_height_fill,
                        annotation_width_fill: annotation_width + annotation_width_fill] = annotation
        save_annotation = Image.fromarray(annotation_fill, mode="P")
        save_annotation.putpalette(pal)
        save_annotation.save(png_trans_path_prefix + line[1])
