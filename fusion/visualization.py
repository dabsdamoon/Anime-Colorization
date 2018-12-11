##### This is a short function for  image visualization  to show colorization results

##### importing functions

import cv2
from skimage.color import rgb2gray, rgb2lab, lab2rgb
import numpy as np
import matplotlib.pyplot as plt

##### Function visualizing 3 iamges (input, original, predicted)

def visualize_validation_image(image_valid, trained_model, classifier, image_size):

    image_valid = cv2.resize(image_valid, (128, 128))
    img_gray = rgb2gray(image_valid).reshape(128, 128, 1)
    img_gray3 = np.concatenate([img_gray]*3, axis = 2) # concatenating three gray images so that it can have 3 channels
    img_lab3 = rgb2lab(img_gray3) # convert concatenated image to lab

    valid_input = img_lab3[:,:,0].reshape(1, image_size[0], image_size[1], 1)/128
    pred = trained_model.predict([valid_input, classifier.predict(valid_input)])    
    pred = pred.reshape(image_size[0], image_size[1], 2)

    cur_pred = np.zeros((image_size[0], image_size[1], 3)) 
            
    # Output colorizations
    cur_pred[:,:,0] = img_lab3[:,:,0]  # lab class
    cur_pred[:,:,1:] = pred*128  # lab predicted

    ##### imshow 3 images (input, original, predicted) as concatenated image

    img_result_concat = np.concatenate([image_valid, 
                                        np.concatenate([img_gray.reshape(image_size[0], image_size[1], 1)]*3, axis = 2), 
                                        lab2rgb(cur_pred)], 
                                       axis = 1)

    plt.figure(figsize = (12, 12))
    plt.imshow(img_result_concat)
    plt.axis('off')
    plt.show()