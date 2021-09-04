CUDA_VISIBLE_DEVICES=9  python3 main.py --pre_train  './experiments/pretrained/model_basic.pt'  \
                        --save test_4k \
                        --test_only \
                        --n_feats 64
                        
# --pre_train './experiments/basic/model/model_latest.pt'
