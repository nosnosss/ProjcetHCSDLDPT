# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import numpy as np
import cv2
from skimage import feature  # Thêm phần import cho HOG features

# Chuyển đổi giá trị màu của một điểm ảnh từ hệ màu RGB sang hệ màu HSV
def rgb_to_hsv(pixel):
    r, g, b = pixel 
    r, g, b = b / 255.0, g / 255.0, r / 255.0
    
    v = max(r, g, b)
    delta = v - min(r, g, b)
    
    if delta == 0:
        h = 0
        s = 0
    else:
        s = delta / v
        if r == v:
            h = (g - b) / delta
        elif g == v:
            h = 2 + (b - r) / delta
        else:
            h = 4 + (r - g) / delta
        h = (h / 6) % 1.0
        
    return [int(h * 180), int(s * 255), int(v * 255)]

# Chuyển đổi một ảnh từ hệ màu RGB sang hệ màu HSV
def covert_image_rgb_to_hsv(img):
    hsv_image = []
    for i in img:
        hsv_image2 = []
        for j in i:
            new_color = rgb_to_hsv(j)
            hsv_image2.append((new_color))
        hsv_image.append(hsv_image2)
    hsv_image = np.array(hsv_image)
    return hsv_image

# Tính toán histogram của một ảnh đầu vào dựa trên các kênh màu được chỉ định, kích thước histogram và khoảng giá trị
def my_calcHist(image, channels, histSize, ranges):
    hist = np.zeros(histSize, dtype=np.int64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            bin_vals = [image[i, j, c] for c in channels]
            bin_idxs = [(bin_vals[c] - ranges[c][0]) * histSize[c] // (ranges[c][1] - ranges[c][0]) for c in range(len(channels))]
            hist[tuple(bin_idxs)] += 1
    return hist

# Trích xuất HSV
data_HSV = []
image_folder = 'static/uploads'
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

for i, image_path in enumerate(image_files):
    img = cv2.imread(image_path)
    bins = [8, 12, 3]
    ranges = [[0, 180], [0, 256], [0, 256]]
    img_hsv = covert_image_rgb_to_hsv(img)
    hist_my = my_calcHist(img_hsv, [0, 1, 2], bins, ranges)
    embedding = hist_my.flatten()
    fixed_image_path = image_path.replace('\\', '/')
    data_HSV.append([fixed_image_path, embedding])
    print(i, end=' ')

np.save("HSV.npy", data_HSV)

# Chuyển đổi ảnh từ hệ màu RGB sang hệ màu xám
def convert_image_rgb_to_gray(img_rgb, resize="no"):
    h, w, _ = img_rgb.shape
    img_gray = np.zeros((h, w), dtype=np.uint32)
    for i in range(h):
        for j in range(w):
            r, g, b = img_rgb[i, j]
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            img_gray[i, j] = gray_value
    if resize != "no":
        img_gray = cv2.resize(src=img_gray, dsize=(496, 496))
    return np.array(img_gray)

# Tính toán đặc trưng HOG (Histogram of Oriented Gradients) của một ảnh xám 
def hog_feature(gray_img):
    (hog_feats, hogImage) = feature.hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                                        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2",
                                        visualize=True)
    return hog_feats

# Trích xuất HOG
data_hog = []
for i, image_path in enumerate(image_files):
    img = cv2.imread(image_path)
    img_gray = convert_image_rgb_to_gray(img)
    embedding = hog_feature(img_gray)
    embedding = embedding.flatten()
    fixed_image_path = image_path.replace('\\', '/')
    data_hog.append([fixed_image_path, embedding])
    print(i, end=' ')

np.save("HOG.npy", data_hog)

data_file_hsv = np.load("HSV.npy", allow_pickle=True)
data_file_hog = np.load("HOG.npy", allow_pickle=True)

array_concat_hog_hsv = []
for i in range(len(data_file_hog)):
    concat_in_value = np.concatenate((data_file_hsv[i][1], data_file_hog[i][1]))
    array_concat_hog_hsv.append([data_file_hog[i][0].replace('\\', '/'), concat_in_value])

np.save("concat_hog_hsv.npy", array_concat_hog_hsv)
