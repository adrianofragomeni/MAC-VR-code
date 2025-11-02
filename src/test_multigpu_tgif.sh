  GNU nano 2.9.8                                            test_multigpu_tgif.sh                                                       

#!/bin/bash

#SBATCH --job-name=training.py
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
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
    --batch_size_val 64
    --anno_path "../data/TGIF/anns/llama3/"
    --video_path "../data/TGIF/videos/"
    --datatype tgif
    --max_words 32
    --max_frames 12
    --video_framerate 1
    --output_dir "../results/TGIF/MAC-VR-qb-6_8_BS128-test/"
    --center 8
    --temp 3
    --alpha 0.01
    --beta 0.005
    --init_model "../results/TGIF/MAC-VR-6-8-BS128/pytorch_model.bin.step3840.6"
    --number_textual_tags_train 6
    --number_textual_tags_test 8
    --number_visual_tags_train 6
    --number_visual_tags_test 8
    --output_path_concepts "../tsne/MAC-VR-TGIF-concepts.json"
"

echo "Start"
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=${MASTER_PORT} --nproc_per_node=1 main_retrieval.py ${ARGS}
echo "Done"

