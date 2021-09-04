CUDA_VISIBLE_DEVICES=9 python3 main.py  \
                              --save basic \
                              --save_model \
          										--dir_test_LR './data/CameraFusion/val/LR/x2' \
          										--dir_test_ref './data/CameraFusion/val/Ref/x2/' \                              
                              --loss 1*L1+0.05*contextual_ref+0.01*contextual_hr \
                              --patch_size 128 \
                              --batch_size 4 \
                              --n_feats 64

