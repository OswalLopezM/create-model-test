from argparse import ArgumentParser
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle



word_dict = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
    10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',
    18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z', 26: 'Ñ'
}

def preprocess():
    image = cv2.imread('./test images/marcador.jpg')
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (24, 24))
    dilated = cv2.dilate(thresh.copy(), kernel)

    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(len(contours))

    preprocessed_digits = []

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if(h >= 70 and h <= 200 and w > 50):
            print("x = %s , y = %s , w = %s , h = %s  "%(x,y,w,h))
            # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
            cv2.rectangle(image, (x+12,y+12), (x+w-12, y+h-12), color=(0, 255, 0), thickness=2)
            
            # Cropping out the digit from the image corresponding to the current contours in the for loop
            digit = thresh[y+12:y+h-12, x+12:x+w-12]
            
            # Resizing that digit to (18, 18)
            resized_digit = cv2.resize(digit.copy(), (18,18),interpolation = cv2.INTER_AREA)
            
            # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
            padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
            
            # Adding the preprocessed digit to the list of preprocessed digits
            # plt.imshow(padded_digit)
            # plt.show()
            preprocessed_digits.append(padded_digit)


    shuff = shuffle(preprocessed_digits)
    fig, ax = plt.subplots(3,3, figsize = (10,10))
    axes = ax.flatten()

    for i in range(9):
        _, shu = cv2.threshold(shuff[i], 30, 200, cv2.THRESH_BINARY)
        axes[i].imshow(np.reshape(shuff[i], (28,28)), cmap="Greys")
    plt.show()
    print("\n\n\n----------------Contoured Image--------------------")
    plt.imshow(image, cmap="gray")
    plt.show()

    inp = np.array(preprocessed_digits)
    return preprocessed_digits

def predict(preprocessed_digits,option_file):

    h5_file = ''
    if(option_file=='mine'):
        h5_file = './models/model_hand.h5'
    if(option_file=='minerva10'):
        h5_file = './models/minerva-10.h5'

    model = tf.keras.models.load_model(h5_file)

    for digit in preprocessed_digits:
        prediction = model.predict(digit.reshape(1, 28, 28, 1))  
        
        print ("\n\n---------------------------------------\n\n")
        print ("=========PREDICTION============ \n\n")
        plt.imshow(digit.reshape(28, 28), cmap="gray")
        print("\n\nFinal Output: {}".format(np.argmax(prediction)))
        
        print ("\nPrediction (Softmax) from the neural network:\n\n {}".format(prediction))
        
        hard_maxed_prediction = np.zeros(prediction.shape)
        hard_maxed_prediction[0][np.argmax(prediction)] = 1

        print("\n\nFinal Output: %s"%format(word_dict[int(np.argmax(prediction))]))
        print ("\n\nHard-maxed form of the prediction: \n\n {}".format(hard_maxed_prediction))
        print ("\n\n---------------------------------------\n\n")
        plt.title(word_dict[int(np.argmax(prediction))])
        plt.show()


def main():

    parser = ArgumentParser(
        description="%(prog)s Predict algorithm"
    )
    parser.add_argument(
        '--file', '-f', 
        help='Choose the h5 file wich is gonna use Example "minervann","minerva10", "mine"', 
        choices=['minerva10','minervann', 'mine'])

    args = parser.parse_args()
    option_file = args.file

    preprocessed_digits = preprocess()
    predict(preprocessed_digits, option_file)

if __name__ == "__main__":
    main()