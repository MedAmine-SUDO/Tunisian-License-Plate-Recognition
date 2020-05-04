import pandas as pd
import urllib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import time
from PIL import Image

from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

PD_TRAIN_IMAGES = "/home/medamine/Desktop/Dataset-pfa/license_plates_detection_train"
CHANNEL = 3

df = pd.read_csv('license_plates_detection_train.csv')

dataset = dict()
dataset["image_name"] = list()
dataset["image_width"] = list()
dataset["image_height"] = list()
dataset["ymin"] = list()
dataset["xmin"] = list()
dataset["ymax"] = list()
dataset["xmax"] = list()   

for index, row in df.iterrows():
    img = Image.open(PD_TRAIN_IMAGES+'/'+row['img_id'])
    dataset["image_name"].append(row['img_id'])
    width, height = img.size 
    dataset["image_width"].append(width)
    dataset["image_height"].append(height)
    dataset["ymin"].append(row['ymin'])
    dataset["xmin"].append(row['xmin'])
    dataset["ymax"].append(row['ymax'])
    dataset["xmax"].append(row['xmax'])

df = pd.DataFrame(dataset)
print(df.head())

lucky_test_samples = np.random.randint(0, len(df), 5)
reduced_df = df.drop(lucky_test_samples, axis=0)

def show_img(index):
    image = cv2.imread(PD_TRAIN_IMAGES+'/'+ df['image_name'].iloc[index])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tx = int(df["xmin"].iloc[index])
    ty = int(df["ymin"].iloc[index])
    bx = int(df["xmax"].iloc[index])
    by = int(df["ymax"].iloc[index])

    image = cv2.rectangle(image, (tx, ty), (bx, by), (0, 0, 255), 1)
    plt.imshow(image)
    plt.show()

