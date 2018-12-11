##### This is a collection of functions I used for processing images.

##### Importing function

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab

##### function just uploading images to Python (RGB transformation and other process will be committed in ImageDataGenerator part later)

def data_preprocessing(dir_name, size_row, size_col):

    train_image = []
    train_label = []

    test_image = []
    test_label = []

    image_size = (size_row, size_col)

    folder_name = os.listdir(dir_name)

    character_name = [i[4:] for i in folder_name]
    character_name.sort()

    folder_dict = {}

    for i in range(len(folder_name)):
        folder_dict[folder_name[i][4:]] = folder_name[i]

    image_dict = {}

    for i in range(len(folder_name)):

        file_list = os.listdir(dir_name + folder_name[i])
        image_list = [i for i in file_list if i[-3:] == 'png']

        image_dict[folder_name[i][4:]] = image_list

    label_dict = {}

    for i in range(len(folder_name)):
        label_dict[folder_name[i][4:]] = i

    label_dict_inv = {v: k for k, v in label_dict.items()}

    for i in range(len(character_name)):

        print(str(i+1) +'/'+str(len(character_name)))

        image_file = image_dict[character_name[i]]


        ##### split into train and test here

        train_index, test_index = train_test_split(np.arange(len(image_file)), test_size = 0.3)

        train_list = np.array(image_file)[list(train_index)]
        test_list = np.array(image_file)[list(test_index)]

        # train case

        for train_num in range(len(train_list)):

            file_chosen = dir_name + '{}/{}'.format(folder_dict[character_name[i]], train_list[train_num])
            image_chosen = cv2.resize(plt.imread(file_chosen), image_size)

            if image_chosen.shape[2] == 3:
                train_image.append(image_chosen)    

        # test case

        for test_num in range(len(test_list)):

            file_chosen = dir_name + '{}/{}'.format(folder_dict[character_name[i]], test_list[test_num])
            image_chosen = cv2.resize(plt.imread(file_chosen), image_size)

            if image_chosen.shape[2] == 3:
                test_image.append(image_chosen)

    train_image = np.array(train_image)
    test_image = np.array(test_image)

    return train_image, test_image

