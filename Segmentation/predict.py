from ultralytics import YOLO
import torch
import cv2
import numpy as np

class Segmentation:
    def __init__(self,model_path=None,mask_threshold=0.5,scale=0.4, bilinear=False):

        
        self.mask_threshold=mask_threshold
        self.scale=scale
        self.bilinear=bilinear
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

        #Declaring Model
        if model_path is None:
            self.model = YOLO("../deep_sort_3D/yolo_v8/runs/detect/train2/weights/best.pt")  # load a custom model
        
        else:
            self.model=YOLO(model_path)

    def predict_img(self,full_img, out_threshold=0.5):

        results = self.model(full_img)  # predict on an image
               
        masks = results[0].masks.data
        boxes = results[0].boxes.data
    
        # extract classes
        clss = boxes[:, 5]
        mask_indices = torch.where(clss == 0)
        class_masks = masks[mask_indices]
        
        class_masks = torch.any(class_masks, dim=0).int() * 255
        binary_mask = class_masks.cpu().numpy().astype(np.uint8)
        binary_mask = cv2.resize(binary_mask, (full_img.shape[1], full_img.shape[0]))
        return binary_mask