CUDA_VISIBLE_DEVICES=9 python3 main.py 	--pre_train  './experiments/pretrained/model_basic.pt' \
					--save test_8k \
					--dir_test_LR './data/CameraFusion/test/HR/' \
					--dir_test_ref './data/CameraFusion/test/Ref/x1/' \
					--test_only  \
					--flag_8k \
					--n_feats 64
										
#--pre_train './experiments/basic/model/model_latest.pt'
