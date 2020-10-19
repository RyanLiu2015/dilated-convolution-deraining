import os
from PIL import Image

os.chdir('dataset\\light_train\\rain\\')

for i in range(1800):
#for i in range(200):
    im = Image.open(str(i+1) + '.png')
    width, height = im.size
    if width != 481:
        im_rotated = im.rotate(90, expand=True)
        im_rotated.save(str(i+1) + '.png')