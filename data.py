import matplotlib.pyplot as plt
import numpy as np
import pathlib
import cv2
from PIL import Image
import os
import re

currentDir = os.path.dirname(os.path.realpath(__file__))

def get_mnist():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels


def get_input():
    images = np.array([[]])
    labels = np.array([[]])
    i = 0

    for filename in  os.listdir(currentDir + '\\data\\testing_data\\'):
        if filename.endswith('.png'):
            file = currentDir + '\\data\\testing_data\\' + filename 
            test_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            test_image = cv2.resize(test_image, (28,28), interpolation=cv2.INTER_LINEAR)
            test_image = cv2.bitwise_not(test_image)
            image, label = get_data(file)
            if i == 0:
                images = np.append(images, [image], axis=1)
                labels = np.append(labels, [label], axis=1)
                i += 1
            else:
                images = np.append(images, [image], axis=0)
                labels = np.append(labels, [label], axis=0)
    return images, labels

            
def get_data(file_directory, return_label=True):
    if return_label:
        img = Image.open(file_directory).convert('L')
        image = np.array(img.getdata())
        image = image / 255

        number = int(file_directory[-5])
        label = np.zeros((10))
        label[number] = 1
        return image, label
    else:
        img = Image.open(file_directory).convert('L')
        image = np.array(img.getdata())
        image = image / 255
        return image


def get_drawing(file_directory):
    img = cv2.imread(file_directory, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28,28), interpolation=cv2.INTER_CUBIC)
    img_array = Image.fromarray(img)
    img_array.save(currentDir + '\\temp\\test.png')
    
    return img

