from PIL import Image


def scale(input_img, size):
    # load pixels of input_img
    input_pixels = input_img.load()
    # load width and height of output_img and input_img
    in_x, in_y = input_img.size
    out_x, out_y = int(size[0]), int(size[1])
    # creat a output_img
    output_img = Image.new('L', (out_x, out_y))
    output_pixels = output_img.load()
    # define lambda for dectectBound
    dectectBound = lambda x, bound: x if x < bound - 1 else x - 1
    # compute the rate of scale
    rateX = (in_x / out_x)
    rateY = (in_y / out_y)
    for i in range(out_x):
        for j in range(out_y):
            tempX = i * rateX
            tempY = j * rateY
            # detect cross-bound
            x = dectectBound(tempX, in_x - 1)
            y = dectectBound(tempY, in_y - 1)
            # get the position of pixels in the input_img
            x1, y1 = int(x), int(y)
            x2, y2 = x1 + 1, y1 + 1
            # bi_linear
            output_pixels[i, j] = int((x2 - x) * (y2 - y) * input_pixels[x1, y1] +
                                      (x - x1) * (y2 - y) * input_pixels[x2, y1] +
                                      (x2 - x) * (y - y1) * input_pixels[x1, y2] +
                                      (x - x1) * (y - y1) * input_pixels[x2, y2])

    return output_img
