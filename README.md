# Yolov1-implementation-using-pytroch
Yolov1 implementation using pytorch with a resnet50 backbone 



The bacbone.py file simply loads the Resnet50 model which acts as the backbone for extracting the convolutional features from the model

The util.py file simple loads the files and consists of  a generator function that provides x_batch (image in a tensor of shape(batch_size , 7 , 7 , 30) the 7,7,30 comes from yolov1 paper specification and y_bacth is also a tensor of (batch_Size , 7 , 7 , 30) i kept the this shape to preserve the spatial information 

the loss.py has the loss function

the train.py simply trains the model
