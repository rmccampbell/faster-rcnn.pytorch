#!/usr/bin/env bash
#SBATCH -o test_faster_rcnn_%j.out
#SBATCH -e test_faster_rcnn_%j.out
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -A class-cs6501-004-sp19

module purge
module load singularity
# module load tensorflow

# Assuming that the container has been copied to the user's /scratch directory
containerdir=/scratch/$USER

singularity exec --nv \
    $containerdir/tensorflow-1.12.0-py36.simg \
    python -u test_net.py --dataset svhn --cuda --net vgg16 "$@"
