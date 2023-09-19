# -----------------------------
### Train SLR
export ratio=$2
export type=$3
export tmp=$4
export data=$5

# ### Step 1: SLR train
export lr=0.01

if [ "$data" = nyu ]; then 
    export baseline_str=NYUv2
elif [ "$data" = cityscape ]; then 
    export baseline_str=CityScapes
fi

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --arch adashare \
    --data $data \
    --slr-prune \
    --admm-train \
    --config-file APIs/mtl_pytorch/profile/$data/config_$ratio\_$tmp.yml \
    --save-dir ./ \
    --sparsity-type $type \
    --baseline checkpoint/$baseline_str.model \
    --epochs 100 \
    --prune-lr $lr \
    --ext _$tmp\lr$lr


### Step 2: Retrain
# export Rlr=0.001
# export Rep=50

export Rlr=0.0001
export Rep=100
# export Rlr=$5

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --arch adashare \
    --data $data \
    --slr-prune \
    --masked-retrain \
    --config-file APIs/mtl_pytorch/profile/$data/config_$ratio\_$tmp.yml \
    --save-dir ./ \
    --sparsity-type $type \
    --slr-model config_$ratio\_$tmp/config_$ratio\_$tmp\_$type\_$tmp\lr$lr.model \
    --retrain-epochs $Rep \
    --prune-lr $Rlr \
    --ext _$tmp\lr$lr\Rlr$Rlr | tee log/$data/retrain/retrain_$ratio\_$type\_$tmp\lr$lr\Rlr$Rlr.txt


# # ------------
# ### Evaluation

# export data=$2

# for ratio in 0.99; do
# for type in channel filter; do
# for tmp in wHS9_all; do

# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --arch adashare \
#     --data $data \
#     --evaluate checkpoint/prune_$data/slr_retrain/config_$ratio\_$tmp/config_$ratio\_$tmp\_$type\_$tmp\lr0.01Rlr0.001.model | tee log/$data/eval/eval_$ratio\_$type\_$tmp\lr0.01Rlr0.001.txt

# done
# done
# done

# for ratio in 0.7; do
# for type in channel filter; do
# for tmp in wH_all; do

# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --arch adashare \
#     --data $data \
#     --evaluate checkpoint/prune_$data/slr_retrain/config_$ratio\_$tmp/config_$ratio\_$tmp\_$type\_$tmp\lr0.01Rlr0.001.model | tee log/$data/eval/eval_$ratio\_$type\_$tmp\lr0.01Rlr0.001.txt

# done
# done
# done


# #------
# ## Evaluate the SLR-trained model

# export ratio=$2
# export type=$3
# export tmp=$4
# export data=$5

# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --arch adashare \
#     --data $data \
#     --config-file APIs/mtl_pytorch/profile/$data/config_$ratio\_$tmp.yml \
#     --sparsity-type $type \
#     --evaluate checkpoint/prune_$data/slr_train/config_$ratio\_$tmp/config_$ratio\_$tmp\_$type\_$tmp\lr0.01.model
#     # --evaluate checkpoint/prune_$data/slr_retrain/config_$ratio\_$tmp/config_$ratio\_$tmp\_$type\_$tmp\lr0.01Rlr0.001.model | tee log/$data/eval/eval_$ratio\_$type\_$tmp\lr0.01Rlr0.001.txt


# #------
# ## Evaluate the Retrained model

# export ratio=$2
# export type=$3
# export tmp=$4
# export data=$5

# CUDA_VISIBLE_DEVICES=$1 python3 main.py \
#     --arch adashare \
#     --data $data \
#     --evaluate checkpoint/prune_$data/slr_retrain/config_$ratio\_$tmp/config_$ratio\_$tmp\_$type\_$tmp\lr0.01Rlr0.001.model # | tee log/$data/eval/eval_$ratio\_$type\_$tmp\lr0.01Rlr0.001.txt

