cd /home/team/zhaohongwei/Code/Research/Model/finetune_model &&
conda activate peft

#nohup ./train.sh > ./res/finetune-last-mlp-inr.out 2>&1 &
#
#nohup ./train.sh > ./res/finetune-mlp-inr-ina-CosineLinear.out 2>&1 &
#
#nohup ./train.sh > ./res/finetune-last-mlp-inr-ina.out 2>&1 &
#
#nohup ./train.sh > ./res/finetune-last-mlp-cifar.out 2>&1 &

#nohup ./train.sh > ./res/finetune-mlp-ina-focal_loss.out 2>&1 &

#nohup ./train.sh > ./res/finetune-mlp-ina-cb_loss.out 2>&1 &

nohup ./train.sh > ./res/finetune-cifar-DDP.out 2>&1 &

CUDA_VISIBLE_DEVICES=3,4 torchrun --standalone --nproc_per_node=2 main.py --config ./exps/cifar.json