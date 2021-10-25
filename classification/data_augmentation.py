"""
The data augmentation method of Group 9.
"""
from skimage import io, transform, filters
from skimage import color, data_dir, data
from skimage import exposure, img_as_float

import numpy as np
import os
import re
import cv2

# CUR_PATH = './x-image2'
#
# SAVE_PATH = './x-gen'
#
# GEN_PATH = './x-gen'


# ----------------------------- methods --------------------------------------------------------------------------------
def img_gray(imgfile):
    rgb = io.imread(imgfile)
    # 将rgb图片转换成灰度图
    gray = color.rgb2gray(rgb)
    return gray


def gray_from_dir(imagepath, savepath):
    # str=imagepath+'\\*.jpg:'+imagepath+'\\*.png'
    str = imagepath + '\\*.jpg'
    coll = io.ImageCollection(str, load_func=img_gray)
    # 循环保存图片
    for i in range(len(coll)):
        file_name = coll.files[i]
        file_name = re.findall('\d+', file_name)[0]
        io.imsave(savepath + "/" + file_name + '_gy' + '.jpg', coll[i])

# ############################
# # PCA 白化 图像
def RGB_PCA(images):
    pixels = images.reshape(-1, images.shape[-1])
    # idx = np.random.random_integers(0, pixels.shape[0], len(pixels))
    idx = np.random.randint(0, pixels.shape[0], len(pixels))
    pixels = [pixels[i] for i in idx]
    pixels = np.array(pixels, dtype=np.uint8).T
    m = np.mean(pixels) / 256.
    C = np.cov(pixels) / (256. * 256.)
    l, v = np.linalg.eig(C)
    return l, v, m


def RGB_variations(image, eig_val, eig_vec):
    a = np.random.randn(3)
    v = np.array([a[0] * eig_val[0], a[1] * eig_val[1], a[2] * eig_val[2]])
    variation = np.dot(eig_vec, v)
    return image + variation


def img_pca(imgfile):
    img = io.imread(imgfile)
    l, v, m = RGB_PCA(img)
    img_pca = RGB_variations(img, l, v)
    return img_pca


def pca_from_dir(imagepath, savepath):
    # str=imagepath+'\\*.jpg'+':'+imagepath+'\\*.png'
    str = imagepath + '\\*.jpg'

    coll = io.ImageCollection(str, load_func=img_pca)

    # 循环保存图片
    for i in range(len(coll)):
        file_name = coll.files[i]
        file_name = re.findall('\d+', file_name)[0]
        io.imsave(savepath + "/" + file_name + '_pca' + '.jpg', coll[i])

# ############################
# # noise 随机加噪
NOISE_NUMBER = 1000

def img_noise(imgfile):
    img = io.imread(imgfile)
    height, weight, channel = img.shape

    # 随机生成5000个椒盐噪声
    for i in range(NOISE_NUMBER):
        x = np.random.randint(0, height)
        y = np.random.randint(0, weight)
        img[x, y, :] = 255

    return img


def noise_from_dir(imagepath, savepath):
    # str=imagepath+'\\*.jpg'+':'+imagepath+'\\*.png'
    str = imagepath + '\\*.jpg'

    coll = io.ImageCollection(str, load_func=img_noise)

    # 循环保存图片
    for i in range(len(coll)):
        file_name = coll.files[i]
        file_name = re.findall('\d+', file_name)[0]
        io.imsave(savepath + "/" + file_name + '_ns' + '.jpg', coll[i])
        # io.imsave(savepath + '\\ns_' + np.str(i) + '.jpg', coll[i])

# ------------------  gaussian blur-
# -=================================================================
def img_gaussian_blur(imgfile):
    img = io.imread(imgfile)
    return img
    # kernel_size = (5, 5)
    # sigma = 1.5
    # blurred_img = filters.gaussian_filter(img, 2, multichannel=True, mode='reflect')

    # img = cv2.imread(imgfile)
    # img = cv2.GaussianBlur(img, kernel_size, sigma)
    # new_imgName = "New_" + str(kernel_size[0]) + "_" + str(sigma) + "_" + imgName
    # cv2.imwrite(new_imgName, img)
    # return blurred_img

def GaussianBlur_from_dir(imagepath, savepath):
    str = imagepath + '\\*.jpg'
    coll = io.ImageCollection(str, load_func=img_gaussian_blur)

    # 循环保存图片
    for i in range(len(coll)):
        file_name = coll.files[i]
        file_name = re.findall('\d+', file_name)[0]
        io.imsave(savepath + "/" + file_name + '_blur' + '.jpg', coll[i])

# # ############################
# # # rotate 旋转图像 逆时针
# from keras.preprocessing import image as ksimage
#
#
# def ks_rotate(x, theta, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
#     rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
#     h, w = x.shape[row_axis], x.shape[col_axis]
#     transform_matrix = ksimage.transform_matrix_offset_center(rotation_matrix, h, w)
#     x = ksimage.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
#     return x
#
#
# def img_rotate0030(imgfile):
#     img = io.imread(imgfile)
#     rotate_limit = (0, 30)
#     theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])  # 逆时针旋转角度
#     # rotate_limit= 30 #自定义旋转角度
#     # theta = np.pi /180 *rotate_limit #将其转换为PI
#     img_rot = ks_rotate(img, theta)
#     return img_rot
#
#
# def img_rotate3060(imgfile):
#     img = io.imread(imgfile)
#     rotate_limit = (30, 60)
#     theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])  # 逆时针旋转角度
#     # rotate_limit= 30 #自定义旋转角度
#     # theta = np.pi /180 *rotate_limit #将其转换为PI
#     img_rot = ks_rotate(img, theta)
#     return img_rot
#
#
# def img_rotate6090(imgfile):
#     img = io.imread(imgfile)
#     rotate_limit = (60, 90)
#     theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])  # 逆时针旋转角度
#     # rotate_limit= 30 #自定义旋转角度
#     # theta = np.pi /180 *rotate_limit #将其转换为PI
#     img_rot = ks_rotate(img, theta)
#     return img_rot
#
#
# def img_rotate90140(imgfile):
#     img = io.imread(imgfile)
#     rotate_limit = (90, 140)
#     theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])  # 逆时针旋转角度
#     # rotate_limit= 30 #自定义旋转角度
#     # theta = np.pi /180 *rotate_limit #将其转换为PI
#     img_rot = ks_rotate(img, theta)
#     return img_rot
#
#
# def img_rotate140180(imgfile):
#     img = io.imread(imgfile)
#     rotate_limit = (140, 180)
#     theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])  # 逆时针旋转角度
#     # rotate_limit= 30 #自定义旋转角度
#     # theta = np.pi /180 *rotate_limit #将其转换为PI
#     img_rot = ks_rotate(img, theta)
#     return img_rot
#
#
# def img_rotate4000(imgfile):
#     img = io.imread(imgfile)
#     rotate_limit = (-40, 0)
#     theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])  # 逆时针旋转角度
#     # rotate_limit= 30 #自定义旋转角度
#     # theta = np.pi /180 *rotate_limit #将其转换为PI
#     img_rot = ks_rotate(img, theta)
#     return img_rot
#
#
# def img_rotate9040(imgfile):
#     img = io.imread(imgfile)
#     rotate_limit = (-90, -40)
#     theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])  # 逆时针旋转角度
#     # rotate_limit= 30 #自定义旋转角度
#     # theta = np.pi /180 *rotate_limit #将其转换为PI
#     img_rot = ks_rotate(img, theta)
#     return img_rot
#
#
# def img_rotate14090(imgfile):
#     img = io.imread(imgfile)
#     rotate_limit = (-140, -90)
#     theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])  # 逆时针旋转角度
#     # rotate_limit= 30 #自定义旋转角度
#     # theta = np.pi /180 *rotate_limit #将其转换为PI
#     img_rot = ks_rotate(img, theta)
#     return img_rot
#
#
# def img_rotate180140(imgfile):
#     img = io.imread(imgfile)
#     rotate_limit = (-180, -140)
#     theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])  # 逆时针旋转角度
#     # rotate_limit= 30 #自定义旋转角度
#     # theta = np.pi /180 *rotate_limit #将其转换为PI
#     img_rot = ks_rotate(img, theta)
#     return img_rot
#
#
# # #rotateflag = 1,2,3,4,5,6,7,8,9
# # # 1[0,30],2[30,60],3[60,90],4[90,140],5[140,180],6[-180,-140],7[-140,-90],8[-90,-40],9[-40,0]
# def rotate_from_dir(imagepath, savepath, rotateflag=2):
#     # str=imagepath+'\\*.jpg'+':'+imagepath+'\\*.png'
#     str = imagepath + '\\*.jpg'
#
#     if (rotateflag > 4):
#         if (rotateflag == 5):
#             coll = io.ImageCollection(str, load_func=img_rotate140180)
#         if (rotateflag == 6):
#             coll = io.ImageCollection(str, load_func=img_rotate180140)
#         if (rotateflag == 7):
#             coll = io.ImageCollection(str, load_func=img_rotate14090)
#         if (rotateflag == 8):
#             coll = io.ImageCollection(str, load_func=img_rotate9040)
#         else:
#             coll = io.ImageCollection(str, load_func=img_rotate4000)
#     else:
#         if (rotateflag == 4):
#             coll = io.ImageCollection(str, load_func=img_rotate90140)
#         if (rotateflag == 3):
#             coll = io.ImageCollection(str, load_func=img_rotate6090)
#         if (rotateflag == 1):
#             coll = io.ImageCollection(str, load_func=img_rotate0030)
#         else:
#             coll = io.ImageCollection(str, load_func=img_rotate3060)
#
#     # 循环保存图片
#     for i in range(len(coll)):
#         io.imsave(savepath + '\\rs_' + np.str(rotateflag) + '_' + np.str(i) + '.jpg', coll[i])

# ----------------------------- generate -------------------------------------------------------------------------------
# gray_from_dir(imagepath="data/train", savepath="data/train_augmented/grey")
# pca_from_dir(imagepath="data/train", savepath="data/train_augmented/pca")
# noise_from_dir(imagepath="data/train", savepath="data/train_augmented/noise")

GaussianBlur_from_dir(imagepath="data/train", savepath="data/train_augmented/blur")


# imgName = "1.jpg"
# kernel_size = (5, 5)
# sigma = 1.5
#
# img = cv2.imread(imgName)
# img = cv2.GaussianBlur(img, kernel_size, sigma)
# new_imgName = "New_" + str(kernel_size[0]) + "_" + str(sigma) + "_" + imgName
# cv2.imwrite(new_imgName, img)
