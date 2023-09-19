CUDA_VISIBLE_DEVICES=1 python3 main.py --save-dir separate --pretrain --ext _separate
CUDA_VISIBLE_DEVICES=1 python3 main.py --save-dir separate --alter-train --pretrain-model pre_train_all_10000iter.model --ext _separate
CUDA_VISIBLE_DEVICES=1 python3 main.py --save-dir separate --post-train --alter-model alter_train_with_reg_0005_20000iter.model --shared 0 --ext _separate

