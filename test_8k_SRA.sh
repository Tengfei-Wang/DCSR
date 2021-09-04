CUDA_VISIBLE_DEVICES=9 python3 main.py --pre_train './experiments/pretrain/model_SRA.pt'  \
										--save test_8k_SRA \
										--dir_test_LR './data/CameraFusion/test/HR/' \
										--dir_test_ref './data/CameraFusion/test/Ref/x1/' \
										--test_only  \
										--flag_8k \
                    								--n_feats 64

#--pre_train './experiments/SRA/model/model_latest.pt'
