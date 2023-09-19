# -------------------------------------------
# ### Train SLR

# export ratio=$2
# export type=$3
# export lr=0.01
# export Rlr=0.001

# # CUDA_VISIBLE_DEVICES=$1 python3 main_mtl.py \
# #     --slr-prune \
# #     --admm-train \
# #     --config-file config_$ratio \
# #     --save-dir new_multi_11 \
# #     --sparsity-type $type \
# #     --baseline post_train_21_shrd0New_dcy2000_30000iter.model \
# #     --epochs 100 \
# #     --prune-lr $lr \
# #     --ext _SpSkw18lr$lr

# CUDA_VISIBLE_DEVICES=$1 python3 main_mtl.py \
#     --slr-prune \
#     --admm-train \
#     --config-file config_$ratio \
#     --save-dir new_multi_11 \
#     --sparsity-type $type \
#     --baseline post_train_21_shrd0New_dcy2000_30000iter.model \
#     --epochs 100 \
#     --prune-lr $lr \
#     --ext _Allw18lr$lr

# -------------------------------------------
# ### Retrain

# # CUDA_VISIBLE_DEVICES=$1 python3 main_mtl.py \
# #     --slr-prune \
# #     --masked-retrain \
# #     --config-file config_$ratio \
# #     --save-dir new_multi_11 \
# #     --sparsity-type $type \
# #     --slr-model config_$ratio/config_$ratio\_$type\_SpSkw18lr0.01_101iter.model \
# #     --retrain-epochs 50 \
# #     --prune-lr $Rlr \
# #     --ext _SpSkw18lr0.01Rlr$Rlr | tee log/retrain/retrain_$ratio\_$type\_SpSkw18lr0.01Rlr$Rlr.txt

# CUDA_VISIBLE_DEVICES=$1 python3 main_mtl.py \
#     --slr-prune \
#     --masked-retrain \
#     --config-file config_$ratio \
#     --save-dir new_multi_11 \
#     --sparsity-type $type \
#     --slr-model config_$ratio/config_$ratio\_$type\_Allw18lr0.01_101iter.model \
#     --retrain-epochs 50 \
#     --prune-lr $Rlr \
#     --ext _Allw18lr0.01Rlr$Rlr | tee log/retrain/retrain_$ratio\_$type\_Allw18lr$lr\Rlr$Rlr.txt


# -----------------------------
# ### evaluate

# for ratio in 0.9 0.99 0.999 div div2; do
#     for type in irregular channel filter column; do
#         echo "$ratio,$type"

#         CUDA_VISIBLE_DEVICES=$1 python3 main_mtl.py \
#             --evaluate checkpoint/new_multi_11/prune/slr_retrain/config_$ratio/config_$ratio\_$type\_Allw18lr0.01Rlr0.001_51iter.model | tee log/eval/eval_$ratio\_$type\_Allw18lr0.01Rlr0.001.model

#     done
# done


# export ratio=$2
# export type=$3

# CUDA_VISIBLE_DEVICES=$1 python3 main_mtl.py \
#     --evaluate checkpoint/new_multi_11/prune/slr_retrain/config_$ratio/config_$ratio\_$type\_Allw18lr0.01Rlr0.001_51iter.model | tee log/eval/eval_$ratio\_$type\_Allw18lr0.01Rlr0.001.model


export path=$2      # model path
CUDA_VISIBLE_DEVICES=$1 python3 main_mtl.py --evaluate $path