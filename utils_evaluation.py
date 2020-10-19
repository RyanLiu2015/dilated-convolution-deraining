from math import log10, sqrt 
from PIL import Image
import cv2
import numpy as np

from SSIM_PIL import compare_ssim

def psnr(): 
    

    original = cv2.imread("eva1.png") 
    compressed = cv2.imread("eva2.png", 1) 
    
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))

    print('psnr:', psnr)  

def ssim():
    image1 = Image.open("eva1.png")
    image2 = Image.open("eva2.png")
    value = compare_ssim(image1, image2)

    print('ssim:', value)

    return


if __name__ == '__main__':
    psnr()
    ssim()
