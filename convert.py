import pandas as pd
from PIL import Image
import xml.etree.ElementTree as ET


df = pd.read_csv('/kaggle/input/plate-zindi/license_plates_detection_train.csv')
IMAGES_TRAINING = "/kaggle/input/plate-zindi/license_plates_detection_train/license_plates_detection_train/"

dataset = dict()
dataset["image_name"] = list()
dataset["image_width"] = list()
dataset["image_height"] = list()
dataset["ymin"] = list()
dataset["xmin"] = list()
dataset["ymax"] = list()
dataset["xmax"] = list()   

for index, row in df.iterrows():
    img = Image.open(IMAGES_TRAINING+row['img_id'])
    dataset["image_name"].append(row['img_id'])
    width, height = img.size 
    dataset["image_width"].append(width)
    dataset["image_height"].append(height)
    dataset["ymin"].append(row['ymin'])
    dataset["xmin"].append(row['xmin'])
    dataset["ymax"].append(row['ymax'])
    dataset["xmax"].append(row['xmax'])

df = pd.DataFrame(dataset)
print(df)

for index, row in df.iterrows():
    root = ET.Element("annotation")
    folder = ET.SubElement(root, 'folder')
    folder.text = IMAGES_TRAINING
    filenameElemnt = ET.SubElement(root, 'filename')
    filenameElemnt.text = row["image_name"]
    sizeElement = ET.SubElement(root, 'size')
    widthElemtn = ET.SubElement(sizeElement, 'width')
    widthElemtn.text = str(row["image_width"])
    heightElement = ET.SubElement(sizeElement, 'height')
    heightElement.text = str(row["image_height"])
    depthElement = ET.SubElement(sizeElement, 'depth')
    depthElement.text = str(3)
    objectElement = ET.SubElement(root, 'object')
    bndboxElement = ET.SubElement(objectElement, 'bndbox')
    xmin = ET.SubElement(bndboxElement, 'xmin')
    xmin.text = str(row["xmin"])
    ymin = ET.SubElement(bndboxElement, 'ymin')
    ymin.text = str(row["ymin"])
    xmax = ET.SubElement(bndboxElement, 'xmax')
    xmax.text = str(row["xmax"])
    ymax = ET.SubElement(bndboxElement, 'ymax')
    ymax.text = str(row["ymax"])

    tree = ET.ElementTree(root)
    tree.write(row["image_name"][:-4]+".xml")
    