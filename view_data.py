import numpy as np

file_path = 'concat_hog_hsv.npy'

data = np.load(file_path, allow_pickle=True)

print(data)
