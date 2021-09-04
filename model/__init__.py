import os
from importlib import import_module
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import imageio
import numpy as np
class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale
        self.flag_8k = args.flag_8k
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half': self.model.half()

        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.model, file=ckp.log_file)

    def forward(self, x, ref):
        target = self.get_model()

        if not self.training:
            if not self.flag_8k:
                min_size1 = 126
                min_size2 = 126           
            else:
                if x.shape[2] < x.shape[3]:
                    min_size1 = 188
                    min_size2 = 252
                else:    
                    min_size1 = 252
                    min_size2 = 188

            x = x[:,:,: x.shape[2]//min_size1*min_size1,:x.shape[3]//min_size2*min_size2]
            shape_query = x.shape
            num_x = x.shape[3]//min_size2
            num_y = x.shape[2]//min_size1
            x = nn.ZeroPad2d(20)(x)
            sr_list = []
            for j in range(num_y):
                for i in range(num_x):
                    patch_LR = x[:,:,j*(min_size1):j*(min_size1) + min_size1 +40, i*min_size2:i*min_size2 + min_size2+40]

                    if (i > (num_x//4 -1)) and (i < (3*num_x//4) ) and (j > (num_y//4 -1)) and (j < (3*num_y//4)  ):
                        patch_ref = ref[:,:,np.maximum((j -  num_y//4)*2*min_size1 - 20, 0):np.minimum((j -  num_y//4)*2*min_size1 + 2*min_size1 +80,ref.shape[2] ), np.maximum((i -  num_x//4)*2*min_size2 -20,0):np.minimum((i -  num_x//4)*2*min_size2 + 2*min_size2 +80 , ref.shape[3])] 
                        coarse = False
                                    
                    elif ((i == (num_x//4 -1))  and (j >= (num_y//4 -1))  and (j < (3*num_y//4)  )) or ((j == (num_y//4 -1))  and (i >= (num_x//4 -1))  and (i < (3*num_x//4  ))):
                        patch_ref = ref[:,:,np.maximum((j -  num_y//4)*2*min_size1, 0):np.maximum((j -  num_y//4)*2*min_size1, 0) + 2*min_size1 +100, np.maximum((i -  num_x//4)*2*min_size2,0):np.maximum((i -  num_x//4)*2*min_size2,0) + 2*min_size2+100]
                        coarse = False         
                        
                    elif ((i ==  (3*num_x//4))  and (j > (num_y//4 -1))  and (j <= (3*num_y//4 ) )) or ((j == (3*num_y//4))  and (i > (num_x//4 -1))  and (i <= (3*num_x//4  ))):
                        patch_ref = ref[:,:,np.minimum((j -  num_y//4)*2*min_size1, ref.shape[2]- 2*min_size1) :np.minimum((j -  num_y//4)*2*min_size1, ref.shape[2]- 2*min_size1)+ 2*min_size1 ,np.minimum((i -  num_x//4)*2*min_size2,ref.shape[3]- 2*min_size2) :np.minimum((i -  num_x//4)*2*min_size2,ref.shape[3]- 2*min_size2)+ 2*min_size2 ]
                        coarse = False        
                                
                    elif (j == (num_y//4 -1)) and (i ==  (3*num_x//4)):
                        patch_ref = ref[:,:,0:2*min_size1+100 , ref.shape[3]- 2*min_size2-100:ref.shape[3]]   
                        coarse = False   
                        
                    elif (j == (3*num_y//4)) and (i ==  (num_x//4 -1)):
                        patch_ref = ref[:,:, ref.shape[2]- 2*min_size1-100:ref.shape[2],0:2*min_size2+100 ]     
                        coarse = False   
                                    
                    else:    
                        patch_ref = ref   
                        coarse = True  
                
                    patch_sr = self.model(patch_LR, patch_ref, coarse)
                    sr_list.append(patch_sr[:,:,40:-40, 40:-40])
            
                
            sr_list = torch.cat(sr_list, dim=0)
            sr_list = sr_list.view(sr_list.shape[0],-1)
            sr_list = sr_list.permute(1,0) 
            sr_list = torch.unsqueeze(sr_list, 0)
            output = F.fold(sr_list, output_size=(shape_query[2]*2, shape_query[3]*2), kernel_size=(2*min_size1,2*min_size2), padding=0, stride=(2*min_size1,2*min_size2))
            return output

        else:
            return self.model(x, ref,False)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch):
        target = self.get_model()
        torch.save(
            target.state_dict(), 
            os.path.join(apath, 'model_latest.pt')
        )

        
        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == -1:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
        elif resume == 0:
            if pre_train != '.':
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=False
                )
        else:
            print('Loading model from {}'.format(os.path.join(apath, 'model', 'model_{}.pt'.format(resume))))
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=False
            )




