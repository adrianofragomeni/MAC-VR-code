#!/bin/bash

#SBATCH --job-name=training.py
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=100GB 


module purge
source activate /user/home/ms20996/miniconda3/envs/DiCoSA

# Dynamically generate a master port using the SLURM job ID to avoid port conflicts
MASTER_PORT=$((2500 + SLURM_JOB_ID % 1000))

ARGS="
    --do_train 0
    --do_eval 1
    --workers 8
    --n_display 50
    --epochs 30
    --lr 1e-4
    --coef_lr 1e-3
    --batch_size 128
    --batch_size_val 32
    --anno_path "../data/YC2/anns/llama3/"
    --video_path "../data/YC2/videos/"
    --datatype yc2
    --max_words 32
    --max_frames 24
    --video_framerate 1
    --output_dir "../results/YC2/MAC-VR-qb-8_10_BS128_24frames-test"
    --center 8
    --temp 3
    --alpha 0.01
    --beta 0.005
    --init_model "../results/YC2/MAC-VR-qb-8_10_BS128_24frames/pytorch_model.bin.step1350.16"
    --number_textual_tags_train 8
    --number_textual_tags_test 10
    --number_visual_tags_train 8
    --number_visual_tags_test 10
    --output_path_concepts "../tsne/MAC-VR-YC2-concepts.json"
"

echo "Start"
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port ${MASTER_PORT} --nproc_per_node=1 main_retrieval.py ${ARGS}
echo "Done"



