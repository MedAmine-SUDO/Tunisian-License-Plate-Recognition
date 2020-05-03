#Import libraries
import pandas
import cv2
import os
import glob

plate_detection_train_images_path = "/home/medamine/Desktop/Dataset-pfa/license_plates_detection_train"
plate_recognition_train_images_path = "/home/medamine/Desktop/Dataset-pfa/license_plates_recognition_train"

lp_detection = pandas.read_csv('license_plates_detection_train.csv')
#print(lp_detection)

plateDetectionImages = [cv2.imread(plate_detection_train_images_path+'/'+file) for file in os.listdir(plate_detection_train_images_path)]

cv2.imshow('image1', plateDetectionImages[1])
cv2.waitKey(0)
cv2.destroyAllWindows()