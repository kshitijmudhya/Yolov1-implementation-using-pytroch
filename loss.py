import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def custom_iou(self , box1 , box2):
        box1x1 = box1[... , 0] - (box1[...,2]/2)
        box1x2 = box1[...,0] + (box1[...,2]/2)
        box1y1 = box1[...,1] - (box1[...,3]/2)
        box1y2 = box1[...,1] + (box1[...,3]/2)

        box2x1 = box2[... , 0] - (box2[...,2]/2)
        box2x2 = box2[...,0] + (box2[...,2]/2)
        box2y1 = box2[...,1] - (box2[...,3]/2)
        box2y2 = box2[...,1] + (box2[...,3]/2)


        x_overlap = torch.maximum( torch.zeros(box1x1.shape[0] , device='cuda') , (torch.minimum(box1x2 , box2x2) - torch.maximum(box1x1 , box2x1)))
        y_overlap = torch.maximum( torch.zeros(box1x1.shape[0] , device='cuda') , (torch.minimum(box1y2 , box2y2) - torch.maximum(box1y1 , box2y1)))
        inter_area = x_overlap * y_overlap

        box1_area = (box1x2 - box1x1) * (box1y2 - box1y1)
        box2_area = (box2x2 - box2x1) * (box2y2 - box2y1)
        union = box1_area + box2_area - inter_area
        return (inter_area / union )
        
    def forward(self , y_true , y_pred):
        obj_mask  =(torch.sum(y_true[...,  :]  ==  1. , dim=3) == 1)
        noobj_mask = (torch.sum(y_true[...,  :] != 1. , dim=3) !=  1)
        
        iou1 = self.custom_iou(y_true[obj_mask][... , :4] , y_pred[obj_mask][... , :4] )
        iou2 = self.custom_iou(y_true[obj_mask][... , :4] , y_pred[obj_mask][... , 6:10] )
        mask = torch.unsqueeze(iou1 > iou2 , dim=1)
        predictor_box = torch.where(mask , y_pred[obj_mask][..., :5] , y_pred[obj_mask][..., 6:11]  )
        nopredictor_box = torch.where(mask ,   y_pred[obj_mask][..., 6:11] , y_pred[obj_mask][..., :5]  )
        classfication = torch.sum(torch.square(y_true[obj_mask][10:]  - y_pred[obj_mask][10:]))
        conf = torch.sum(torch.square(predictor_box[... , 4] - 1)) + torch.sum(torch.square(nopredictor_box[... , 4] - 0))
        conf  += 0.5 * torch.sum(torch.square(y_pred[noobj_mask][..., 4] - 0 )) + 0.5 * torch.sum(torch.square(y_pred[noobj_mask][..., 4] - 0 ))
        try:
            localization = torch.sum(torch.square(predictor_box[... , 0:2] - y_true[... , 0:2])) +  torch.sum(torch.square(predictor_box[... , 2:4] - y_true[... , 2:4]))
        except:
            localization  = 0
        
        return 5*localization + conf + classfication
        