import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import cv2   
import numpy as np                                  

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes                                                                                                                     
                                                                                                                                               
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):                                          
    def __init__(self, dim, vis, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.vis = vis                                        
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
              
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
                                        
        qkv = self.to_qkv(x).chunk(3, dim = -1)                         
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        #print("00000_q:", q.shape)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        #print("1111_dots:", dots.shape)                           

        attn = self.attend(dots)                          
        weights = attn if self.vis else None                            
        #print("####____weights:", weights.shape)        
        attn = self.dropout(attn)                                                        
                                                                
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), weights 

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, vis, dropout = 0.):
        super().__init__()
        self.vis = vis  
        self.norm = nn.LayerNorm(dim)
        """
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
        """  
        self.depth = depth
        self.attn = Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, vis = self.vis)   
        self.ff = FeedForward(dim, mlp_dim, dropout = dropout)        

    def forward(self, x):
        #for attn, ff in self.layers:
        attn_weights = []
        for _ in range(self.depth):            
            out, weights = self.attn(x) 
            if self.vis:                               
                attn_weights.append(weights)                                     
            x = out + x                                                          
            x = self.ff(x) + x                                 
        #print("1111_attn_weights:", len(attn_weights))                          
        return self.norm(x), attn_weights                            
                                                                   
class ViT(nn.Module):                                                 
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, vis, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()                                           
        self.vis = vis                                               
                   
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )                      
                                                                              
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, self.vis)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)
                                                 
    def forward(self, img):              
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)             
        x, attn_weights = self.transformer(x)                                     
        #x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #print("@@@@@@@@@@__#############_$$$$$$$$$$:", x, x.shape)
        if self.pool == 'mean':                                 
            x = x.mean(dim = 1)                                                    
        else:                                                 
            x = x[:, 0]                                                                      
       # print("1111_@@@@@@@@@@__#############_$$$$$$$$$$:", x, x.shape)            
        x = self.to_latent(x)                           
        #print("xxxxx:", x)                                                                    
                                                                                                                                                     
        return self.mlp_head(x), attn_weights                                              