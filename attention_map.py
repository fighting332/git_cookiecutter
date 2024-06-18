

from __future__ import print_function
                                                                                     
import glob
from itertools import chain
import os
import random
import zipfile
           
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms                                                              
from tqdm.notebook import tqdm   
from PIL import Image  
import numpy   
import conf              
                                                                            
#from vit_pytorch.efficient import ViT                                            
from vit_pytorch.vit_attn_rollout import ViT                              
import copy         
import cv2                                                                              
                                                                           
# mlflow               
from mlflow import log_artifacts, log_metric, log_param, log_image                 
                                                                              
                                                                            
# Implement attention map in terms of weight                                    
def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))                                                      
    with torch.no_grad():                                                           
        for attention in attentions:                           
            if head_fusion == "mean":  # mean is the best mode for visualization                             
                attention_heads_fused = attention.mean(axis=1)                                                        
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)                                  
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)                                                      
            indices = indices[indices != 0]                                                                              
            flat[0, indices] = 0                                                   
                                                         
            I = torch.eye(attention_heads_fused.size(-1))               
            attention_heads_fused = attention_heads_fused.cpu()                                                          
            a = (attention_heads_fused + 1.0*I)/2                                                                 
            a = a / a.sum(dim=-1)           

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]                                                                 
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    

# Implement visualization of color space
def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)                                          

                                                    
 
torch.cuda.current_device()
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                          

test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(240),
        transforms.ToTensor(),
    ]
)                                                                        
                                       
                        
if __name__ == "__main__":                                                                   
    image = Image.open(conf.data_path)                                           
    test_data = test_transforms(image)                                                           
                                                                       
    # VIT parameter                           
    model = ViT(             
        image_size = 240,                                         
        patch_size = 20,                           
        num_classes = 2,                                                  
        dim = 128,                             
        depth = 6,                                             
        heads = 16,                   
        mlp_dim = 2048,                             
        dropout = 0.1,                                             
        emb_dropout = 0.1,                               
        vis = True       
    ).to(device)                                                                                                        
                                                                                                                        
    model_path = conf.model_path                                                                                                                 
    checkpoint = torch.load(os.path.join(model_path), map_location=lambda storage, loc: storage)                                                                                                  
    model.load_state_dict(checkpoint)                                                                         
                                                       
    for param in model.parameters():         
        param.grad = None                                              
    model.eval()                                                                
    for k, v in model.named_parameters():                                             
        v.requires_grad = False                                    
                                                                                                                                                                                                                                                                                                                                        
    data = test_data.to(device).unsqueeze(0)                                                                                                                                    
    val_output, att_mat = model(data)                                                           
    val = val_output.argmax(dim=1)                                                                                                                          
    mask = rollout(att_mat, conf.discard_ratio, conf.attention_mode)                                              
                                                                        
    # overray image                                                                                                                                                                                                                                                                                      
    img = Image.open(os.path.join(conf.data_path))               
    img = img.resize((240, 240))                                                                  
    np_img = np.array(img)[:, :, ::-1]                                                                
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)                                                                                                    
    cv2.imshow("Input Image", np_img)                                                          
    cv2.imshow("222", mask)                                                                                           
    cv2.imwrite(os.path.join(conf.save_image, "355_ouput.jpg"), mask)    
                             
    # mlflow: parameter & metric                                                                                                 
    log_param("discard_ratio", conf.discard_ratio)                                
    log_param("attention_mode", conf.attention_mode)                                                    
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)                                                                                                           
    log_image(mask, "attention_map.jpg")                                                                                  
                                                                                             
                                                                          
                                                                               