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
        obj_mask  =(y_true[...,4]   ==  1)
        noobj_mask =(y_true[...,4] !=  1)
        iou1 = self.custom_iou(y_true[obj_mask][... , :4] , y_pred[obj_mask][... , :4] )
        iou2 = self.custom_iou(y_true[obj_mask][... , :4] , y_pred[obj_mask][... , 5:9] )
        mask = (iou1 > iou2).unsqueeze(1).float()
        
        predictor_box =mask *  y_pred[obj_mask][..., :5]  +  (1-mask) * y_pred[obj_mask][..., 5:10]  
        classfication = torch.sum(torch.square(y_true[obj_mask][... , 10:]  - y_pred[obj_mask][... , 10:]))
        conf = torch.sum(torch.square(predictor_box[... , 4] - 1)) 
        conf  += 0.5 * torch.sum(torch.square(y_pred[noobj_mask][..., 4] - 0 )) +0.5 * torch.sum(torch.square(y_pred[noobj_mask][..., 9] - 0 ))
        if predictor_box.shape[0] == 0:
            loc_loss = torch.tensor(0.0, device=y_true.device)
        else:
            xy_loss = torch.sum((predictor_box[..., 0:2] - y_true[obj_mask][..., 0:2]) ** 2)
            wh_loss = torch.sum((torch.sqrt(predictor_box[..., 2:4].clamp(min=1e-6)) -
                                torch.sqrt(y_true[obj_mask][..., 2:4].clamp(min=1e-6))) ** 2)
            loc_loss = xy_loss + wh_loss

        return 5*loc_loss + conf + classfication
        
if __name__ == "__main__":
    loss = Loss()
    loss(torch.ones((32, 7,7,30) , device='cuda') , torch.ones((32, 7,7,30) , device='cuda'))