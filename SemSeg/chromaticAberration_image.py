from ast import main
from nis import match
from sys import argv
from unittest import case
import cv2
import numpy as np
import os, glob



_img_extension = '.png'
_out_extension = 'chroma'

def _chromaticAberration(img):
    return

def _apply_chromaticAberration(img_path):
    img_name = os.path.basename(img_path)
    img_folder = os.path.dirname(img_path)
    if (os.path.isfile(img_path)):
        img = cv2.imread(img_path)
        img_chroma = _chromaticAberration(img)
        cv2.imwrite(img_chroma, os.path.join())
    else:
        raise ValueError('Invalid image path: {}'.format(img_path))
    return

def _chroma_folder(input_folder, output_folder):
    for img_path in glob.glob(input_folder + '/*' + _img_extension):
        _apply_chromaticAberration(img_path)
    return

if __name__ == "__main__":

    img_dir = argv[1]
    
    if os.path.isdir(img_dir):
        output_nb = 1
        output_dir = os.path.join(img_dir,'output_{}'.format(_out_extension))
        while os.path.exists(output_dir):
            output_dir = os.path.join(img_dir,'output_{}_{}'.format(_out_extension, output_nb))
            output_nb +=1
    else:
        raise ValueError("Image input folder does not exist: {}".format(img_dir))    

    os.mkdir(output_dir)
    with open(os.path.join(output_dir, 'chromaticAberration_config.txt'), 'w') as cfg:
        cfg.write(
            '** ChromaticAberration Configuration **\n'
            + 'img_dir = {}\n'.format(img_dir)
            )
        
    _chroma_folder(img_dir, output_dir)