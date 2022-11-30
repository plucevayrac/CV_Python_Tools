from ast import main
from nis import match
from sys import argv
from unittest import case
import cv2
import numpy as np
import os, glob

_kernel = (11,11)
_sigmaX = 1
_sigmaY = 1
_ks = 5
_d = 5
_sigmaColor = 50
_sigmaSpace = 50

_img_extension = '.png'

def _blur(img, kernel):
    return cv2.blur(img, kernel)

def _gaussianBlur(img, kernel, sigmaX, sigmaY):
    return cv2.GaussianBlur(img, kernel, sigmaX, sigmaY)

def _medianBlur(img, ks):
    return cv2.medianBlur(img, ks)

def _bilateralFilter(img, d, sigmaColor, sigmaSpace):
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

def _blur_image(img, blur_type):
    match blur_type:
        case "blur":
            ret = _blur(img, _kernel)
        case "gaussian":
            ret = _gaussianBlur(img, _kernel, _sigmaX, _sigmaY)
        case "median":
            ret = _medianBlur(img, _ks)
        case "bilateral":
            ret = _bilateralFilter(img, _d, _sigmaColor, _sigmaSpace)
    return ret

def _blur_folder(img_dir, output_dir, blur_type):
    for filename in glob.glob(img_dir+'/*'+_img_extension):
        print("Processing image: {}".format(filename))
        img = cv2.imread(filename)
        img = _blur_image(img, blur_type)
        output_path = os.path.join(output_dir, os.path.basename(filename))
        cv2.imwrite(output_path, img)


if __name__ == "__main__":

    img_dir, blur_type = argv[1], argv[2]
    
    if os.path.isdir(img_dir):
        output_nb = 1
        output_dir = os.path.join(img_dir,'output_{}'.format(blur_type))
        while os.path.exists(output_dir):
            output_dir = os.path.join(img_dir,'output_{}_{}'.format(blur_type, output_nb))
            output_nb +=1
    else:
        raise ValueError("Image input folder does not exist: {}".format(img_dir))    

    os.mkdir(output_dir)
    with open(os.path.join(output_dir, 'blur_config.txt'), 'w') as cfg:
        cfg.write(
            '** Blur Configuration **\n'
            + 'img_dir = {}\n'.format(img_dir)
            + 'blur_type = {}\n'.format(blur_type)
            + 'kernel_size = {}\n'.format(_kernel)
            + 'sigmaX = {}\n'.format(_sigmaX)
            + 'sigmaY = {}\n'.format(_sigmaY)
            + 'ks = {}\n'.format(_ks)
            + 'd = {}\n'.format(_d)
            + 'sigmaColor = {}\n'.format(_sigmaColor)
            + 'sigmaSpace = {}\n'.format(_sigmaSpace)
            )
        
    _blur_folder(img_dir, output_dir, blur_type)