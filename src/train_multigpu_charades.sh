#!/bin/bash

#SBATCH --job-name=training.py
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=15
#SBATCH --mem=80GB 


module --force purge
source activate /user/home/ms20996/miniconda3/envs/DiCoSA

# Dynamically generate a master port using the SLURM job ID to avoid port conflicts
MASTER_PORT=$((2500 + SLURM_JOB_ID % 1000))


ARGS="
    --do_train 1
    --do_eval 0
    --workers 8
    --n_display 30
    --epochs 40
    --lr 1e-4
    --coef_lr 1e-3
    --batch_size 128
    --batch_size_val 32
    --anno_path "../data/Charades-STA/anns/llama3/"
    --video_path "../data/Charades-STA/videos/"
    --datatype charades
    --max_words 64
    --max_frames 32
    --video_framerate 1
    --output_dir "../results/Charades-STA/MAC-VR-12-12-BS128-REAL128"
    --center 8
    --temp 3
    --alpha 0.01
    --beta 0.005
    --number_textual_tags_train 12
    --number_textual_tags_test 12
    --number_visual_tags_train 12
    --number_visual_tags_test 12
"

echo "Start"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port ${MASTER_PORT} --nproc_per_node=4 main_retrieval.py ${ARGS}
echo "Done"



