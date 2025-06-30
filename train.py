import torch
import torch.nn as nn
import os
from utils import train_generator
from loss import Loss
from model import Model
def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
device = 'cuda'
epochs = 100
total_images   = len(os.listdir('/home/kshitij/Desktop/facerecognition/VOC2012_train_val/JPEGImages'))
batch_size = 8
steps_per_epoch = total_images // batch_size
data = train_generator(batch_size)
loss_function = Loss()

model = Model().to(device)
optimizer = torch.optim.SGD(params=model.parameters() , lr=0.001 ,  momentum=0.9,weight_decay=0.0005)
for i in range(epochs):
    print(f"The epoch number {i+1}")
    avg_loss = 0
    for j in range(steps_per_epoch):
        x_bacth  , y_batch = next(data)
        model.train()
        y_pred = model(x_bacth)
        optimizer.zero_grad()
        loss = loss_function(y_batch , y_pred)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        print(f"The loss for epoch {i+1} and step {j} is {avg_loss/(j+1)}")
    if i == 0 : 
            set_lr(optimizer, 0.01)
    torch.save(model.state_dict(), f"yolo_pytorch_epoch{i+1}.pt")