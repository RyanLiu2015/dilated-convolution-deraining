import cv2
import os
from PIL import Image


# concatenate 9 image into one
def concatenate_images(prefix='output_', output_name='concat.png'):
    im_list = []
    for i in range(9):
        im_list.append(Image.open(prefix + str(i+1) + '.png'))
        #im_list.append(Image.open(str(i+1) + '.png'))

    output = Image.new('RGB', (3 * 481, 3 * 321))

    for i in range(9):
        output.paste(im_list[i], (int(i%3)*481, int(i/3)*321))

    output.save(output_name)

# crop an image into 9 images of same size
def crop_image(input_name='target.jpg'):

    img = Image.open(input_name)
    width, height = img.size

    # cut positions
    x0 = int(0)
    x1 = int(1.0 * width / 3)
    x2 = int(1.0 * width / 3 * 2)
    x3 = int(width)

    y0 = int(0)
    y1 = int(1.0 * height / 3)
    y2 = int(1.0 * height / 3 * 2)
    y3 = int(height)

    # (left, top, right, bottom)
    output = img.crop((x0, y0, x1, y1))
    output.save('crop_1.png')
    output = img.crop((x1, y0, x2, y1))
    output.save('crop_2.png')
    output = img.crop((x2, y0, x3, y1))
    output.save('crop_3.png')

    output = img.crop((x0, y1, x1, y2))
    output.save('crop_4.png')
    output = img.crop((x1, y1, x2, y2))
    output.save('crop_5.png')
    output = img.crop((x2, y1, x3, y2))
    output.save('crop_6.png')

    output = img.crop((x0, y2, x1, y3))
    output.save('crop_7.png')
    output = img.crop((x1, y2, x2, y3))
    output.save('crop_8.png')
    output = img.crop((x2, y2, x3, y3))
    output.save('crop_9.png')

    return

# resize 9 croped target images to desired size 321*481 (NOT CROPPED)
def resize_single_image(image_name='target.jpg'): 

    width = 481
    height = 321
    dim = (width, height)

    img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite('resized_' + image_name, resized)

    return

# resize 9 croped target images to desired size 321*481 (NOT CROPPED)
def resize_target_image(): 

    image_prefix= 'crop_'

    width = 481
    height = 321
    dim = (width, height)

    for i in range(9):
        img = cv2.imread(image_prefix + str(i+1) + '.png', cv2.IMREAD_UNCHANGED)
        #print('Original Dimensions : ', img.shape)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        #print('Resized Dimensions : ', resized.shape)
        cv2.imwrite(str(i+1)+'.png', resized)

    return


if __name__ == '__main__':
    #crop_images()
    #resize_target_image()
    concatenate_images()