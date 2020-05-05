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

PD_TRAIN_IMAGES = "/home/medamine/Desktop/Dataset-pfa/license_plates_detection_train/"
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

lucky_test_samples = np.random.randint(0, len(df), 5)
reduced_df = df.drop(lucky_test_samples, axis=0)

WIDTH = 224
HEIGHT = 224

def show_img(index):
    image = cv2.imread(PD_TRAIN_IMAGES+'/'+ df['image_name'].iloc[index])
    width, height = image.shape[0], image.shape[1]
    image = cv2.resize(image, dsize=(WIDTH, HEIGHT))
    x_scale = ( WIDTH / height)
    y_scale = ( HEIGHT / width )
    tx = int(df["xmin"].iloc[index] * x_scale)
    ty = int(df["ymin"].iloc[index]* y_scale)
    bx = int(df["xmax"].iloc[index]* x_scale)
    by = int(df["ymax"].iloc[index]* y_scale)

    image = cv2.rectangle(image, (tx, ty), (bx, by), (0, 0, 255), 1)
    plt.imshow(image)
    plt.show()


datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

train_generator = datagen.flow_from_dataframe(
    reduced_df,
    directory=PD_TRAIN_IMAGES,
    x_col="image_name",
    y_col=["ymin", "xmin", "ymax", "xmax"],
    target_size=("image_width", "image_height"),
    batch_size=32, 
    class_mode=None,
    subset="training")

validation_generator = datagen.flow_from_dataframe(
    reduced_df,
    directory=PD_TRAIN_IMAGES,
    x_col="image_name",
    y_col=["ymin", "xmin", "ymax", "xmax"],
    target_size=("image_width", "image_height"),
    batch_size=32, 
    class_mode=None,
    subset="validation")

model = Sequential()
model.add(VGG16(weights="imagenet", include_top=False, input_shape=(HEIGHT, WIDTH, CHANNEL)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(4, activation="sigmoid"))

model.layers[-6].trainable = False

#model.summary()

#STEP_SIZE_TRAIN = int(np.ceil(train_generator.n / train_generator.batch_size))
#STEP_SIZE_VAL = int(np.ceil(validation_generator.n / validation_generator.batch_size))

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VAL=validation_generator.n//validation_generator.batch_size

print("Train step size:", STEP_SIZE_TRAIN)
print("Validation step size:", STEP_SIZE_VAL)

train_generator.reset()
validation_generator.reset()

adam = Adam(lr=0.0005)
model.compile(optimizer=adam, loss="mse")

history = model.fit_generator(train_generator,
                                steps_per_epoch=STEP_SIZE_TRAIN,
                                validation_data=validation_generator,
                                validation_steps=STEP_SIZE_VAL,
                                epochs=10)