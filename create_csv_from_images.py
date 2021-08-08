from PIL import Image
import numpy as np
import sys
import os
import csv

from tqdm import tqdm

#Useful function
def createFileList(myDir, format='.png'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

# load the original image
myFileList = createFileList('./dataset/images_dataset/F')

for file in tqdm(myFileList):
    # print(file)
    img_file = Image.open(file)
    # img_file.show()

    # get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Make image Greyscale
    img_grey = img_file.convert('L')
    #img_grey.save('result.png')
    #img_grey.show()

    # Save Greyscale values
    value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((28,28))
    value = value.flatten()
    # print(value)
    with open("dataset_f_by_me.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)
