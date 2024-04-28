## 工具函数模块
import os
import pickle

import numpy as np
from PIL import Image
import PIL
import os
import matplotlib.pyplot as plt

def Z_score_normalization(X):
    """
    desc:Z-score归一化
        公式：x* = ( x − μ ) / σ
    paremeters:
        X   np.array (m,n) 原始数据
    return:
        X_nor np.array (m,n)  归一化后的
    """
    # 计算样本的特征的均值和标准差
    Mu =    np.mean(X, axis=0)
    Sigma = np.std(X,  axis=0)

    # print(f"Mu = {Mu}")
    X_nor = (X - Mu) / Sigma

    return X_nor, Mu, Sigma

def min_max_normalize(data):
    """
    对数据进行最大最小值归一化处理
    :param data: 待归一化的数据，可以是列表、数组等可迭代对象
    :return: 归一化后的数据
    """
    min_val = np.max(data, axis=0)
    max_val = np.min(data, axis=0)

    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return np.array(normalized_data)


def check_dimension(array, expected_dimension, column_vector=False):
    """
    检测 NumPy 数组的维度是否符合预期。

    参数：
    - array: 要检测的 NumPy 数组。
    - expected_dimension: 期望的维度。可以是一个整数，表示精确匹配；也可以是一个元组，表示在范围内。
    - column_vector: 是否期望数组是列向量。默认为 False。

    返回：
    - 如果数组的维度符合预期，则返回 True；否则返回 False。
    """
    array = np.array(array)
    actual_dimension = array.ndim

    if isinstance(expected_dimension, int):
        if column_vector:
            return actual_dimension == 2 and array.shape[1] == 1
        else:
            return actual_dimension == expected_dimension
    elif isinstance(expected_dimension, tuple):
        min_dimension, max_dimension = expected_dimension
        if column_vector:
            return min_dimension <= actual_dimension <= max_dimension and array.shape[1] == 1
        else:
            return min_dimension <= actual_dimension <= max_dimension
    else:
        raise ValueError("Invalid expected_dimension. It should be an integer or a tuple.")


def crop_image(image_path, new_size):
    """
    裁剪图像大小。

    参数：
    - image_path: 图像文件路径。
    - new_size: 新的图像大小，格式为 (width, height)。

    返回：
    - 裁剪后的图像对象。
    """
    # 打开图像文件
    image = Image.open(image_path).convert('L')

    # 裁剪图像
    cropped_image = image.resize(new_size)

    return np.array(cropped_image)


def traverse_folder_for_files(folder_path):
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # 将文件路径添加到列表中
            all_files.append(file_path)
    print(all_files)
    return all_files


def rgb_to_gray(rgb_image):
    # 加权平均法转换为灰度图像
    gray_image = np.dot(rgb_image[...,:3], [0.2989, 0.5870, 0.1140])
    return gray_image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def gray_image(X_data=None):
    # 灰度化处理
    # 将图像数据转换为灰度图像
    gray_images = []
    for image in X_data:
        rgb_image = np.reshape(image, (3, 32, 32)).transpose(1, 2, 0)  # 将图像转换为形状为 (32, 32, 3)
        gray_image = rgb_to_gray(rgb_image)  # 转换为灰度图像
        gray_images.append(gray_image.flatten())
    return gray_images