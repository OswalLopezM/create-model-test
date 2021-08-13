import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from argparse import ArgumentParser

parser = ArgumentParser(
        description="%(prog)s Sourcing Stores Automatic Reposition Arguments"
    )
parser.add_argument(
    '--file', '-f', 
    help='Choose the type of reposition to calculate being "atc", "not-atc" and "all" as options', 
    choices=['basic', 'mine'])

args = parser.parse_args()
option_file = args.file

dataset_file = "./dataset/archive/A_Z Handwritten Data/A_Z Handwritten Data.csv" if option_file == 'basic' else "./dataset/archive/A_Z Handwritten Data/dataset_by_me.csv"
data = pd.read_csv(r"./dataset/archive/A_Z Handwritten Data/A_Z Handwritten Data.csv").astype('float32')
# print(dataset_file)
# print(data.head(10))
# print(data.shape[0])

X = data.drop('0',axis = 1)
y = data['0']
print(y)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2)

print(train_x.shape[0])
print(len(train_x.values))
# train_x = np.reshape(train_x.values, (train_x.shape[0], 28,28))

train_x = np.reshape(train_x.values, (train_x.shape[0], 28,28))
##############
shuff = shuffle(train_x[:100])
print("shuff")
print(len(shuff))

fig, ax = plt.subplots(3,3, figsize = (10,10))
axes = ax.flatten()

for i in range(9):
    _, shu = cv2.threshold(shuff[i], 30, 200, cv2.THRESH_BINARY)
    axes[i].imshow(np.reshape(shuff[i], (28,28)), cmap="Greys")
plt.show()