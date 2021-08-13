from argparse import ArgumentParser
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import tensorflow as tf
from sklearn.utils import shuffle
import logging
logging.basicConfig(
format="%(asctime)s%(levelname)s - %(message)s", level=logging.INFO
)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

        print("r %s, heigth: %s, h: %s "%(r, height, h))

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    print((dim))
    resized = cv2.resize(image, dim)

    # return the resized image
    return resized

def order_contours(contours):
    h_list=[]
    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if w*h>250:
            h_list.append([x,y,w,h])
    #print h_list          
    ziped_list=zip(*h_list)
    x_list=list(ziped_list[0])
    dic=dict(zip(x_list,h_list))
    x_list.sort()
    return x_list

def preprocess(image_number):
    logging.info('|||||||||||||||||||||||||||||||||||||||||||||\n')
    logging.info('./new_data_set/nn%s.jpg'%image_number)
    image = cv2.imread('./new_data_set/nn%s.jpg'%image_number)
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (24, 24))
    dilated = cv2.dilate(thresh.copy(), kernel)

    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    preprocessed_digits = []

    index = 0
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        # print("x = %s , y = %s , w = %s , h = %s  "%(x,y,w,h))
        if(h >= 70 and h <= 200 and w > 50):
            # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)

            color = (0, 255, 0)
            cv2.rectangle(image, (x+12,y+12), (x+w-12, y+h-12), color=color, thickness=2)
            
            # Cropping out the digit from the image corresponding to the current contours in the for loop
            digit = thresh[y+12:y+h-12, x+12:x+w-12]

            # Resizing that digit to (18, 18)
            resized_digit = cv2.resize(digit.copy(), (18,18),interpolation = cv2.INTER_AREA)
            padding_width = (5,5)

            # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
            padded_digit = np.pad(resized_digit, ( (5,5) , padding_width), "constant", constant_values=0)
            
            # Adding the preprocessed digit to the list of preprocessed digits
            preprocessed_digits.append(padded_digit)

    shuff = shuffle(preprocessed_digits)
    fig, ax = plt.subplots(3,3, figsize = (10,10))
    axes = ax.flatten()
    print(len(axes))
    for i in range(len(axes)):
        _, shu = cv2.threshold(shuff[i], 30, 200, cv2.THRESH_BINARY)
        print(shuff[i].shape)
        axes[i].imshow(np.reshape(shuff[i], (28,28)), cmap="Greys")
        # axes[i].imshow((shuff[i]), cmap="Greys")
    plt.show()
    print("\n\n\n----------------Contoured Image--------------------")

    plt.imshow(image, cmap="gray")
    plt.show()

    # can_continue = input("save letters? \n")

    can_continue = "y"
    logging.info("%s images to save"%len(preprocessed_digits))
    logging.info('|||||||||||||||||||||||||||||||||||||||||||||\n')

    if can_continue == "y":

        start = image_number*1000
        for index, image in enumerate(preprocessed_digits):
            
            # plt.imshow(image, cmap="gray")
            # plt.show()
            # can_save = input("save?")
            # if can_save == "":
            # print("saved")
            cv2.imwrite("./new_data_set/nn/nn-"+str(index+start)+".png", image)   

    return preprocessed_digits


def main():
    preprocess(1)
    preprocess(2)
    preprocess(3)
    preprocess(4)
    preprocess(5)
    preprocess(6)
    preprocess(7)
    preprocess(8)

if __name__ == "__main__":
    main()