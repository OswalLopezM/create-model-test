import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

import shutil
import os
from pathlib import Path
from sklearn.utils import shuffle

target_test_dir = "./dataset/images_dataset_train_test/test"

Path(target_test_dir+"/N").mkdir(parents=True, exist_ok=True)

file_names = os.listdir(target_test_dir+"/N")



images = []
index = 0
for filename in shuffle(os.listdir(target_test_dir+"/N")):
    img = cv2.imread(os.path.join(target_test_dir+"/N", filename))
    if img is not None:
        index += 1
        img = cv2.resize(img.copy(), (28,28))
        # img = np.reshape(img, (28,28,3))
        images.append(img)

    if index == 10:
        break


fig, ax = plt.subplots(3,3, figsize = (10,10))
axes = ax.flatten()
for i in range(9):
    _, shu = cv2.threshold(images[i], 30, 200, cv2.THRESH_BINARY)
    print(images[i].shape)
    axes[i].imshow(np.reshape(images[i], (28,28,3)), cmap="Greys")
    # axes[i].imshow((images[i]), cmap="Greys")
plt.show()

