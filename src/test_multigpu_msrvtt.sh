#!/bin/bash

#SBATCH --job-name=training.py
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=100GB 


module --force purge
source activate /user/home/ms20996/miniconda3/envs/DiCoSA

# Dynamically generate a master port using the SLURM job ID to avoid port conflicts
MASTER_PORT=$((2500 + SLURM_JOB_ID % 1000))


ARGS="
    --do_train 0
    --do_eval 1
    --workers 4
    --n_display 50
    --epochs 10
    --lr 1e-4
    --coef_lr 1e-3
    --batch_size 128
    --batch_size_val 32
    --anno_path "../data/MSR-VTT/anns/llama3/"
    --video_path "../data/MSR-VTT/videos/"
    --datatype msrvtt
    --max_words 32
    --max_frames 12
    --video_framerate 1
    --output_dir "../results/MSR-VTT/MAC-VR-qb-6_8_BS128-test/"
    --center 8
    --temp 3
    --alpha 0.01
    --beta 0.005
    --init_model "../results/MSR-VTT/MAC-VR-6_8_BS128-llama3/pytorch_model.bin.step7500.5"
    --number_textual_tags_train 6
    --number_textual_tags_test 8
    --number_visual_tags_train 6
    --number_visual_tags_test 8
    --output_path_concepts "../tsne/MAC-VR-MSRVTT-concepts.json"
"

echo "Start"
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=${MASTER_PORT} --nproc_per_node=1 main_retrieval.py ${ARGS}
echo "Done"



