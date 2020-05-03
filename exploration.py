#Import libraries
import pandas
import cv2
import os
import glob

plate_detection_train_images_path = "/home/medamine/Desktop/Dataset-pfa/license_plates_detection_train"
plate_recognition_train_images_path = "/home/medamine/Desktop/Dataset-pfa/license_plates_recognition_train"

lp_detection = pandas.read_csv('license_plates_detection_train.csv')
print(lp_detection.head())

plateDetectionImages = {file: cv2.imread(plate_detection_train_images_path+'/'+file) for file in os.listdir(plate_detection_train_images_path)}

cv2.imshow('image1', plateDetectionImages['2.jpg'])
cv2.waitKey(0)
cv2.destroyAllWindows()

ymin, xmin, ymax, xmax = lp_detection.iloc[1][1], lp_detection.iloc[1][2], lp_detection.iloc[1][3], lp_detection.iloc[1][4]
print(ymin, xmin, ymax, xmax)
image = cv2.rectangle(plateDetectionImages['10.jpg'],(xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()