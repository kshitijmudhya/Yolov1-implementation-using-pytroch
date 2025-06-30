import torch
import torch.nn as nn
import torchvision.io as io
import os 
import numpy as np
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((448, 448)),  
    transforms.ToTensor()           
])


def preprocess(path , path_pass = True):
    if path_pass:
        image_tensor = Image.open(path).convert('RGB')
        image_tensor = transform(image_tensor)
    else:
        image_tensor = Image.fromarray(path)
        image_tensor = transform(image_tensor)
    return image_tensor
def get_label(path):
    tree = ET.parse(path)
    root = tree.getroot()
    label = np.zeros(( 7 , 7 ,30) )
    VOC_CLASSES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]
    for object_box in root.findall('object'):
        name = object_box.find('name')
        name = name.text
        bndbox = object_box.find('bndbox')
        width = float(root.find('size').find('width').text)
        height = float(root.find('size').find('height').text)
        xmin = float(bndbox.find('xmin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        ymin = float(bndbox.find('ymin').text)
        x_center = (xmax + xmin)/2
        y_center = (ymax+ymin)/2
        width_box = abs(xmax-xmin) / width
        height_box = abs(ymax-ymin) /height
        one_hot = np.zeros(len(VOC_CLASSES))
        class_idx = VOC_CLASSES.index(name)
        one_hot[class_idx] = 1
        grid_row = int((y_center / height) * 7)
        grid_row = max(0, min(grid_row, 6))  
        grid_col = int((x_center / width) * 7)
        grid_col = max(0, min(grid_col, 6))  
        x_center_normalized = ((x_center / width) * 7) - grid_col
        y_center_normalize = ((y_center / height) * 7) - grid_row
        
        label[ grid_row  , grid_col  , :5] = [x_center_normalized , y_center_normalize , width_box , height_box  , 1.] 
        label[grid_row, grid_col, 5:10] = [x_center_normalized , y_center_normalize , width_box , height_box  , 1.] 


        label[grid_row , grid_col , 10:] = one_hot


        
        
    return label
def train_generator(batch_size):
    image_list =os.listdir('/home/kshitij/Desktop/facerecognition/VOC2012_train_val/JPEGImages/') 
   
    while True:
        idx = np.arange(len(image_list))
        np.random.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i:i+batch_size]
            X_batch = [preprocess('/home/kshitij/Desktop/facerecognition/VOC2012_train_val/JPEGImages/' + image_list[j]) for j in batch_idx]
            y_batch = [get_label('/home/kshitij/Desktop/facerecognition/VOC2012_train_val/Annotations/' + image_list[j][:len(image_list[j])-3] + 'xml') for j in batch_idx]
            yield torch.tensor(np.array(X_batch) , device='cuda'), torch.tensor(np.array(y_batch) , device='cuda')

if __name__ == "__main__":
    pass