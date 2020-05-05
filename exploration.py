# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# This script is about exploring the training and test set
# You need to know these things about training/test set:
#     - shape and size
#     - type of the data and type of labels 
# Next you need to display 10 images(trainig) with their repesctive labels 

# %%
#Import libraries
import pandas
import cv2
import os
import glob

plate_detection_train_images_path = "/home/medamine/Desktop/Dataset-pfa/license_plates_detection_train"
plate_recognition_train_images_path = "/home/medamine/Desktop/Dataset-pfa/license_plates_recognition_train"


# %%
lp_detection = pandas.read_csv('license_plates_detection_train.csv')
#lp_detectino columns are: ['img_id', 'ymin', 'xmin', 'ymax', 'xmax']
#convert dataframe to dictionary
lp_detection = lp_detection.set_index('img_id').T.to_dict('list')   
#sorted the dictionary
lp_detection = {k: v for k, v in sorted(lp_detection.items(), key=lambda item: item[1])}


# %%
plateDetectionImages = {file: cv2.imread(plate_detection_train_images_path+'/'+file) for file in os.listdir(plate_detection_train_images_path)}


# %%
cv2.imshow('image1', plateDetectionImages['2.jpg'])
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%
l = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg', '10.jpg']
for item in l:
    xmin = lp_detection[item][1]
    ymin = lp_detection[item][0]
    xmax = lp_detection[item][3]
    ymax = lp_detection[item][2]

    image = cv2.rectangle(plateDetectionImages[item],(xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
    cv2.imshow(item, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# %%


