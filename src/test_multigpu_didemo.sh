#!/bin/bash

#SBATCH --job-name=training.py
#SBATCH --partition=mlcnu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=35:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=15
#SBATCH --mem=80GB 
#SBATCH --account=cosc028886


module purge
source activate /user/home/ms20996/miniconda3/envs/DiCoSA

ARGS="
    --do_train 0
    --do_eval 1
    --workers 8
    --n_display 50
    --epochs 10
    --lr 1e-4
    --coef_lr 1e-3
    --batch_size 64
    --batch_size_val 16
    --anno_path "../data/DiDemo/anns/"
    --video_path "../data/DiDemo/videos/"
    --datatype didemo
    --max_words 64
    --max_frames 50
    --video_framerate 1
    --output_dir "../results/DiDemo/MAC-VR-qb-12_12_BS64-test/"
    --center 8
    --temp 3
    --alpha 0.01
    --beta 0.005
    --init_model "../results/DiDemo/MAC-VR-qb-12_12_BS64/pytorch_model.bin.4"
    --number_textual_tags_train 12
    --number_textual_tags_test 12
    --number_visual_tags_train 12
    --number_visual_tags_test 12
    --output_path_concepts "../tsne/MAC-VR-DiDemo-concepts.json"
"

echo "Start"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 2502 --nproc_per_node=4 main_retrieval.py ${ARGS}
echo "Done"



