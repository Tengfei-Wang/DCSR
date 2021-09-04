import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
from PIL import Image
from torchvision import transforms
from torchvision import utils as vutils
from torchvision import models
from model.common import *
from utils.tools import *
from model.alignment import AlignedConv2d

class FeatureMatching(nn.Module):
    def __init__(self, ksize=3, k_vsize=1,  scale=2, stride=1, in_channel =3, out_channel =64, conv=default_conv):
        super(FeatureMatching, self).__init__()
        self.ksize = ksize
        self.k_vsize = k_vsize
        self.stride = stride  
        self.scale = scale
        match0 =  BasicBlock(conv, 128, 16, 1,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True))

        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.feature_extract = torch.nn.Sequential()
        
        for x in range(7):
            self.feature_extract.add_module(str(x), vgg_pretrained_features[x])
            
        self.feature_extract.add_module('map', match0)
        
   
        for param in self.feature_extract.parameters():
            param.requires_grad = True
            

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 , 0.224, 0.225 )
        self.sub_mean = MeanShift(1, vgg_mean, vgg_std) 
        self.avgpool = nn.AvgPool2d((self.scale,self.scale),(self.scale,self.scale))            


    def forward(self, query, key,flag_8k):
        #input query and key, return matching
    
        query = self.sub_mean(query)
        if not flag_8k:
           query  = F.interpolate(query, scale_factor=self.scale, mode='bicubic',align_corners=True)
        # there is a pooling operation in self.feature_extract
        query = self.feature_extract(query)
        shape_query = query.shape
        query = extract_image_patches(query, ksizes=[self.ksize, self.ksize], strides=[self.stride,self.stride], rates=[1, 1], padding='same') 
      

        key = self.avgpool(key)
        key = self.sub_mean(key)
        if not flag_8k:
           key  = F.interpolate(key, scale_factor=self.scale, mode='bicubic',align_corners=True)
        # there is a pooling operation in self.feature_extract
        key = self.feature_extract(key)
        shape_key = key.shape
        w = extract_image_patches(key, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride], rates=[1, 1], padding='same')
    

        w = w.permute(0, 2, 1)   
        w = F.normalize(w, dim=2) # [N, Hr*Wr, C*k*k]
        query  = F.normalize(query, dim=1) # [N, C*k*k, H*W]
        y = torch.bmm(w, query) #[N, Hr*Wr, H*W]
        relavance_maps, hard_indices = torch.max(y, dim=1) #[N, H*W]   
        relavance_maps = relavance_maps.view(shape_query[0], 1, shape_query[2], shape_query[3])      

        return relavance_maps,  hard_indices


class AlignedAttention(nn.Module):
    def __init__(self,  ksize=3, k_vsize=1,  scale=1, stride=1, align =False):
        super(AlignedAttention, self).__init__()
        self.ksize = ksize
        self.k_vsize = k_vsize
        self.stride = stride
        self.scale = scale
        self.align= align
        if align:
          self.align = AlignedConv2d(inc=128, outc=1, kernel_size=self.scale*self.k_vsize, padding=1, stride=self.scale*1, bias=None, modulation=False)        

    def warp(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, lr, ref, index_map, value):
        # value there can be features or image in ref view

        # b*c*h*w
        shape_out = list(lr.size())   # b*c*h*w
 
        # kernel size on input for matching 
        kernel = self.scale*self.k_vsize

        # unfolded_value is extracted for reconstruction 

        unfolded_value = extract_image_patches(value, ksizes=[kernel, kernel],  strides=[self.stride*self.scale,self.stride*self.scale], rates=[1, 1], padding='same') # [N, C*k*k, L]
        warpped_value = self.warp(unfolded_value, 2, index_map)
        warpped_features = F.fold(warpped_value, output_size=(shape_out[2]*2, shape_out[3]*2), kernel_size=(kernel,kernel), padding=0, stride=self.scale) 
        if self.align:
          unfolded_ref = extract_image_patches(ref, ksizes=[kernel, kernel],  strides=[self.stride*self.scale,self.stride*self.scale], rates=[1, 1], padding='same') # [N, C*k*k, L]
          warpped_ref = self.warp(unfolded_ref, 2, index_map)
          warpped_ref = F.fold(warpped_ref, output_size=(shape_out[2]*2, shape_out[3]*2), kernel_size=(kernel,kernel), padding=0, stride=self.scale)         
          warpped_features = self.align(warpped_features,lr,warpped_ref)        

        return warpped_features     
   

class PatchSelect(nn.Module):
    def __init__(self,  stride=1):
        super(PatchSelect, self).__init__()
        self.stride = stride             

    def forward(self, query, key):
        shape_query = query.shape
        shape_key = key.shape
        
        P = shape_key[3] - shape_query[3] + 1 #patch number per row
        key = extract_image_patches(key, ksizes=[shape_query[2], shape_query[3]], strides=[self.stride, self.stride], rates=[1, 1], padding='valid')

        query = query.view(shape_query[0], shape_query[1]* shape_query[2] *shape_query[3],1)

        y = torch.mean(torch.abs(key - query), 1)

        relavance_maps, hard_indices = torch.min(y, dim=1, keepdim=True) #[N, H*W]   
        

        return  hard_indices.view(-1), P, relavance_maps

