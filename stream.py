import torch
import torch.nn as nn
from model import Model
import cv2
import numpy as np
from utils import preprocess

def draw(x1 ,y1 , x2, y2 , path):
    image = path
    x1 /= 448
    y1 /= 448
    x2 /= 448
    y2 /= 448   
    height, width = image.shape[:2]
    cv2.rectangle(image , (int(x1*width) , int(y1 * height)), (int(x2 * width) , int(y2 * height)) , (0,252,0),2)
    return image
device = 'cuda'
model =  Model().to(device)
model.load_state_dict(torch.load("final.pt" , weights_only=True))

maximum = 0 
grid_row = 0
grid_col  = 0
def draw_box(prediction , path):
    maximum = 0 
    for i in range(7):
        for j in range(7):
            b1 = prediction[0][i , j ,4]
            b2 = prediction[0][i , j ,9]
            if max(b1 , b2)  > maximum:
                maximum = max(b1  , b2 )
                grid_row = i 
                grid_col  = j
    if prediction[0][grid_row , grid_col , 4] == maximum:
        x , y , w, h = prediction[0 , grid_row ,grid_col  , :4]
        x = (grid_col  + x) * 64
        y = (grid_row + y) * 64
        w = w *  448
        h =h * 448
        image = draw(x-(w/2) , y-(h/2)  ,x+(w/2) , y+(h/2) , path )
        
    else:
        x , y , w, h = prediction[0 , grid_row ,grid_col  , 6:10] 
        
        x = (grid_col  + x) * 64
        y = (grid_row + y) * 64 
        w  = w *  448
        h = h * 448
        image = draw(x-(w/2) , y-(h/2)  ,x+(w/2) , y+(h/2) , path )
    onehot = prediction[0 , grid_row , grid_col , 10:]
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
    
    print(VOC_CLASSES[torch.argmax(onehot).to('cpu').numpy()])
    return image


cap = cv2.VideoCapture(0)       
 
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    original_frame  =frame
    # If frame is read correctly, ret is True
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    frame = preprocess(frame  , False  ).to(device)
        
    prediction = model(frame.reshape((-1, 3, 448,448)))
    image = draw_box(  prediction , original_frame)
            
    #print(p
    # Load an image
    


        # Display the resulting frame
    cv2.imshow('Webcam Stream', image)
  
    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()