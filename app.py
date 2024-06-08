import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from skimage import feature

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Hàm kiểm tra đuôi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Hàm chuyển đổi RGB sang HSV
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

# Hàm chuyển đổi ảnh từ RGB sang HSV
def covert_image_rgb_to_hsv(img):
    hsv_image = []
    for i in img:
        hsv_image2 = []
        for j in i:
            new_color = rgb_to_hsv(j)
            hsv_image2.append(new_color)
        hsv_image.append(hsv_image2)
    hsv_image = np.array(hsv_image)
    return hsv_image

# Hàm tính histogram
def my_calcHist(image, channels, histSize, ranges):
    hist = np.zeros(histSize, dtype=np.int64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            bin_vals = [image[i, j, c] for c in channels]
            bin_idxs = [(bin_vals[c] - ranges[c][0]) * histSize[c] // (ranges[c][1] - ranges[c][0]) for c in range(len(channels))]
            hist[tuple(bin_idxs)] += 1
    return hist

# Hàm chuyển đổi RGB sang Gray
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

# Hàm tính HOG feature
def hog_feature(gray_img):
    (hog_feats, hogImage) = feature.hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                                        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2",
                                        visualize=True)
    return hog_feats

# Tải metadata
metadata = np.load('concat_hog_hsv.npy', allow_pickle=True)
metadata = [[data[0].replace('\\', '/'), data[1]] for data in metadata]  # Chuyển đổi đường dẫn sang định dạng đúng

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Kiểm tra xem file có hợp lệ không
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            filepath = filepath.replace('\\', '/')
            file.save(filepath)

            # Xử lý ảnh tải lên
            img = cv2.imread(filepath)
            img_hsv = covert_image_rgb_to_hsv(img)
            bins = [8, 12, 3]
            ranges = [[0, 180], [0, 256], [0, 256]]
            hist_hsv = my_calcHist(img_hsv, [0, 1, 2], bins, ranges).flatten()

            img_gray = convert_image_rgb_to_gray(img)
            hog_feats = hog_feature(img_gray).flatten()

            # Ghép các đặc trưng lại với nhau
            features = np.concatenate((hist_hsv, hog_feats))

            # Tính khoảng cách Euclidean giữa đặc trưng của ảnh tải lên và các ảnh trong metadata
            distances = []
            for data in metadata:
                stored_features = data[1]
                dist = np.linalg.norm(features - stored_features)
                distances.append((data[0], dist))

            # Lấy 3 ảnh có khoảng cách nhỏ nhất
            distances = sorted(distances, key=lambda x: x[1])
            top3 = distances[:3]

            # Tính phần trăm giống nhau
            similarities = [(item[0], 100 * (1 - item[1] / np.max([dist[1] for dist in distances]))) for item in top3]

            return render_template('index.html', filename=filename, results=similarities)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
