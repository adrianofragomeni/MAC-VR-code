#!/bin/bash

#SBATCH --job-name=training.py
#SBATCH --partition=mlcnu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=36:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=15
#SBATCH --mem=100GB 
#SBATCH --account=cosc028886


module purge
source activate /user/home/ms20996/miniconda3/envs/DiCoSA

# Dynamically generate a master port using the SLURM job ID to avoid port conflicts
MASTER_PORT=$((2500 + SLURM_JOB_ID % 1000))

ARGS="
    --do_train 1
    --do_eval 0
    --workers 8
    --n_display 10
    --epochs 30
    --lr 1e-4
    --coef_lr 1e-3
    --batch_size 128
    --batch_size_val 64
    --anno_path "../data/YC2/anns/llama3/"
    --video_path "../data/YC2/videos/"
    --datatype yc2
    --max_words 32
    --max_frames 12
    --video_framerate 1
    --output_dir "../results/YC2/MAC-VR-6_8_BS128_12frames"
    --center 8
    --temp 3
    --alpha 0.01
    --beta 0.005
    --number_textual_tags_train 6
    --number_textual_tags_test 8
    --number_visual_tags_train 6
    --number_visual_tags_test 8
"

echo "Start"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port ${MASTER_PORT} --nproc_per_node=4 main_retrieval.py ${ARGS}
echo "Done"



