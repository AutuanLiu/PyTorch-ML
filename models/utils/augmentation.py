#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  data augmentation example in PyTorch with Augmentor
    1. http://augmentor.readthedocs.io/en/master/userguide/usage.html
    Email : autuanliu@163.com
    Date：2018/04/05
"""
import Augmentor
from pathlib import PurePath

data_dir = PurePath('../../datasets/antsbees/train')

def aug(data_dir, nsample=None, save_format='JPEG'):
    """ image augmentation
    
    Parameters:
    ----------
    data_dir: object of PurePath
        the directory of original image dataset is stored.
    nsample: int
        specify the number of images you require.
    
    Returns:
    --------
        augmented images will by default be saved into an directory named output, 
        relative to the directory which contains your initial image data set.
        And returns the pipeline as a function that can be used with torchvision.
    
    Examples:
    ---------
    >>> from torchvision import transforms
    >>> tfs = transforms.Compose([
    >>>     aug(data_dir),
    >>>     transforms.ToTensor(),
    >>> ])
    """
    p = Augmentor.Pipeline(data_dir)

    # some operation examples
    # p.black_and_white(0.5, threshold=255)
    # horizontal flip
    p.flip_left_right(probability=0.4)
    # vertical flip
    p.flip_top_bottom(probability=0.8)
    # rotate
    p.rotate270(probability=0.5)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.crop_random(probability=1, percentage_area=0.5)
    p.zoom(probability=0.3, min_factor=1.1, max_factor=1.6)

    # sampler, comment this line if you don't want to save images in disk.
    if nsample is not None:
        p.sample(nsample)

    return p.torch_transform()

# example
# x = aug(data_dir, 1000)
# y = aug(data_dir)
