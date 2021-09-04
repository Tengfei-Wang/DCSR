import torch.nn as nn
import torch.nn.functional as F
import torch
from model.attention import *
import matplotlib.pyplot as plt
import numpy as np
import imageio
from model.common import *

def make_model(args, parent=False):
    return DCSR(args)
    

class DCSR(nn.Module):
    def __init__(self, args, conv= default_conv):
        super(DCSR, self).__init__()

   
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale    
        self.flag_8k = args.flag_8k
        # define head module
        m_head1 = [BasicBlock(conv, args.n_colors, n_feats, kernel_size,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True)),
        BasicBlock(conv,n_feats, n_feats, kernel_size,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True))]

        m_head2 = [BasicBlock(conv, n_feats, n_feats, kernel_size,stride=2,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True)),
            BasicBlock(conv, n_feats, n_feats, kernel_size,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True))]            

        # define tail module
        m_tail = [conv3x3(n_feats, n_feats//2), nn.LeakyReLU(0.2, inplace=True), conv3x3(n_feats//2, args.n_colors) ]

        fusion1 = [BasicBlock(conv, 2*n_feats, n_feats, 5,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True)) ,
        BasicBlock(conv, n_feats, n_feats, kernel_size,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True))]

        fusion2= [BasicBlock(conv, 2*n_feats, n_feats, 5,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True)) ,
        BasicBlock(conv, n_feats, n_feats, kernel_size,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True))]        

          
        self.feature_match = FeatureMatching(ksize=3,  scale=2, stride=1, in_channel = args.n_colors, out_channel = 64)

        self.ref_encoder1 = nn.Sequential(*m_head1) #encoder1
        self.ref_encoder2 = nn.Sequential(*m_head2) #encoder3
      
        self.res1 = ResList(4, n_feats) #res3
        self.res2 = ResList(4, n_feats) #res1
        
        self.input_encoder = Encoder_input(8, n_feats, args.n_colors)        

        self.fusion1 = nn.Sequential(*fusion1)
        self.decoder1 = ResList(8, n_feats)     

        
        self.fusion2 = nn.Sequential(*fusion2) #fusion3
        self.decoder2 = ResList(4, n_feats)    #decoder3

        self.decoder_tail = nn.Sequential(*m_tail)

        
        fusion11 = [BasicBlock(conv, 1, 16, 7,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True)) ,
        BasicBlock(conv, 16, n_feats, kernel_size,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True))]
        
        fusion12= [BasicBlock(conv, 1, 16, 7,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True)) ,
        BasicBlock(conv, 16, n_feats, kernel_size,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True))]    
            
        fusion13= [BasicBlock(conv, 4, 32, 7,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True)) ,
        BasicBlock(conv, 32, 3, kernel_size,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True))]  
                 
        self.alpha1 = nn.Sequential(*fusion11)  # g layer in paper
        self.alpha2 = nn.Sequential(*fusion12) # alpha3
        self.alpha3 = nn.Sequential(*fusion13) # alpha4      

        if self.flag_8k:
            self.aa1 = AlignedAttention(scale=4, align=True) #swap4
            self.aa2 = AlignedAttention(scale=2, align = False) #swap3
            self.aa3 = AlignedAttention(scale=4, align=True) #swap2
        else:
            self.aa1 = AlignedAttention(scale=2, align=True) #swap4
            self.aa2 = AlignedAttention(scale=1, align = False) #swap3
            self.aa3 = AlignedAttention(scale=2, align=True) #swap2
        
        self.avgpool = nn.AvgPool2d((2,2),(2,2)) 
        self.select = PatchSelect()
        
    def forward(self,input, ref, coarse = False):
	    
        with torch.no_grad():
          if coarse:

            B = input.shape[0]
            ref_ = F.interpolate(ref, scale_factor=1/16, mode='bicubic')
            input_ = F.interpolate(input, scale_factor=1/8, mode='bicubic')
    
            i , P, r= self.select(input_, ref_)
            
            for j in range(B):
              ref_p = ref[:, :,np.maximum((2*8*(i[j]//P)).cpu(),0):np.minimum((2*8*(i[j]//P)+2*input.shape[2]).cpu(), ref.shape[2]), np.maximum((2*8*(i[j]%P)).cpu(),0):np.minimum((2*8*(i[j]%P)+2*input.shape[3]).cpu(), ref.shape[3])]

          else: 
            ref_p = ref    
        
     
        confidence_map,  index_map = self.feature_match(input, ref_p,self.flag_8k)
       
        ref_downsampled = self.avgpool(ref_p)
        ref_hf = ref_p -  F.interpolate(ref_downsampled, scale_factor=2, mode='bicubic')   #extract high freq in ref
        ref_hf_aligned = self.aa1(input, ref_p, index_map, ref_hf)          
       
        ref_features1 = self.ref_encoder1(ref_p) 
        ref_features1 = self.res1(ref_features1)
   
        ref_features2 = self.ref_encoder2(ref_features1)   
        ref_features2 = self.res2(ref_features2) 
        
        input_down =  F.interpolate(input, scale_factor=1/2, mode='bicubic')
        ref_features_matched = self.aa2(input_down, ref_p, index_map, ref_features2)                  
        ref_features_aligned = self.aa3(input, ref_p, index_map, ref_features1)

        input_up = self.input_encoder(input)

        if  self.flag_8k:
             confidence_map = F.interpolate(confidence_map, scale_factor=2, mode='bicubic')
        cat_features = torch.cat((ref_features_matched, input_up), 1)
        fused_features2 = self.alpha1(confidence_map) * self.fusion1(cat_features) + input_up  #adaptive fusion in feature space
        fused_features2 = self.decoder1(fused_features2)
        fused_features2_up = F.interpolate(fused_features2, scale_factor=2, mode='bicubic')

        confidence_map = F.interpolate(confidence_map, scale_factor=2, mode='bicubic')
        cat_features = torch.cat((ref_features_aligned, fused_features2_up), 1)
        fused_features1 = self.alpha2(confidence_map) *self.fusion2(cat_features) + fused_features2_up #adaptive fusion in feature space
        fused_features1 = self.decoder2(fused_features1)
        result = self.decoder_tail(fused_features1) + ref_hf_aligned*self.alpha3(torch.cat((confidence_map, ref_hf_aligned), 1)) #adaptive fusion in image space

        return result

