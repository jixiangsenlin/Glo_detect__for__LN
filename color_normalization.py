import zipfile
import os, shutil
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image
import time
from mpl_toolkits.mplot3d import Axes3D
import copy
matplotlib.rcParams['figure.figsize'] = (20, 17)
import scipy.signal
import cv2
import torch
import math


# function to do histogram matching
def get_histogram_matching_lut(h_input, h_template):
    ''' h_input: histogram to transfrom, h_template: reference'''
    if len(h_input) != len(h_template):
        # print('histograms length mismatch!')
        return False

    # >> YOUR CODE HERE <<
    LUT = np.zeros(len(h_input))
    H_input = np.cumsum(h_input)  # Cumulative distribution of h_input
    H_template = np.cumsum(h_template)  # Cumulative distribution of h_template

    for i in range(len(H_template)):
        input_index = H_input[i]
        new_index = (np.abs(H_template - input_index)).argmin()
        LUT[i] = new_index

    return LUT, H_input, H_template

def stain_normalization(input_img, target_img, n_bins=100):
    """ Stain normalization based on histogram matching. """

    # print("Lowest value in input_img:" + str(np.min(input_img)))
    # print("Highest value in input_img:" + str(np.max(input_img)))
    #
    # print("Lowest value in target_img:" + str(np.min(target_img)))
    # print("Highest value in target_img:" + str(np.max(target_img)))

    normalized_img = np.zeros(input_img.shape)

    # input_img = input_img.astype(float)  # otherwise we get a complete yellow image
    # target_img = target_img.astype(float)  # otherwise we get a complete blue image
    input_img = input_img.astype('float32')  # otherwise we get a complete yellow image
    target_img = target_img.astype('float32')  # otherwise we get a complete blue image

    # Used resource: https://stackoverflow.com/a/42463602
    # normalize input_img
    input_img_min = input_img.min(axis=(0, 1), keepdims=True)
    input_img_max = input_img.max(axis=(0, 1), keepdims=True)
    if input_img_min[0][0][0] == input_img_max[0][0][0] or input_img_min[0][0][1] == input_img_max[0][0][1] or input_img_min[0][0][2] == input_img_max[0][0][2]:
        return input_img
    # if input_img_min.all() == input_img_max.all():
    #     return input_img
    input_img = (input_img - input_img_min)#input_norm       / (input_img_max - input_img_min)
    input_img = input_img/(input_img_max - input_img_min)

    # normalize target_img
    target_img_min = target_img.min(axis=(0, 1), keepdims=True)
    target_img_max = target_img.max(axis=(0, 1), keepdims=True)
    target_img = (target_img - target_img_min) #target_norm      / (target_img_max - target_img_min)
    target_img = target_img/(target_img_max - target_img_min)


    # Go through all three channels
    for i in range(3):
        input_hist = np.histogram(input_img[:, :, i], bins=np.linspace(0, 1, n_bins + 1))#input_norm
        target_hist = np.histogram(target_img[:, :, i], bins=np.linspace(0, 1, n_bins + 1))#target_norm
        LUT, H_input, H_template = get_histogram_matching_lut(input_hist[0], target_hist[0])
        normalized_img[:, :, i] = LUT[(input_img[:, :, i] * (n_bins - 1)).astype(int)]#input_norm

    normalized_img = normalized_img / n_bins

    return normalized_img#.astype('float32')

def img(HE1, HE2):
    # load data
    # transform HE1 to match HE2
    HE1 = np.float32(HE1)
    HE2 = np.float32(HE2)
    HE1_norm = stain_normalization(HE1, HE2)  # HE1, HE2  HE1_cuda, HE2_cuda
    # HE1_norm = HE1_norm * 255
    # # 缩小
    # h_img = HE1_norm.shape[-3]
    # w_img = HE1_norm.shape[-2]
    # alpha = h_img / w_img
    # t = 30000  # 32767
    # if h_img > t and w_img > t:
    #     if h_img > w_img:
    #         h_img = t
    #         w_img = h_img / alpha
    #     else:
    #         w_img = t
    #         h_img = w_img * alpha
    # elif h_img > t and w_img < t:
    #     h_img = t
    #     w_img = h_img / alpha
    # elif h_img < t and w_img > t:
    #     w_img = t
    #     h_img = w_img * alpha
    #
    # m = int(w_img) * int(h_img)
    # if m >= 89478485:  # 178956970  89478485
    #     print(filename + "  change")
    #     s = math.sqrt(80000000 / m)  # 160000000  80000000
    #     w_img = int(w_img * s)
    #     h_img = int(h_img * s)
    #
    # HE1_norm = cv2.resize(HE1_norm, (int(w_img), int(h_img)))
    return HE1_norm

def normalization(HE1,HE2):
    # HE2 = cv2.imread("131143-2.JPG")
    img1 = img(HE1, HE2)
    return img1

# if __name__ == 'main':
#
#     HE1 = cv2.imread("131143-2.JPG")
#     HE2 = cv2.imread("131143-2.JPG")
#     img1 = img(HE1,HE2)

