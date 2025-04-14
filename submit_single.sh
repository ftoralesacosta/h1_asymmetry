#!/bin/bash
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -n 1
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-task 1
#SBATCH -t 12:00:00
#SBATCH -A m3246
#SBATCH --gpu-bind=none 
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=ftoralesacosta@lbl.gov
##SBATCH --module=gpu,nccl-2.15
module load tensorflow

export HOROVOD_GPU_BROADCAST=NCCL
export HOROVOD_GPU_ALLGATHER=NCCL
export HOROVOD_GPU_ALLREDUCE=NCCL
export MPICH_ALLGATHERV_PIPELINE_MSG_SIZE=0
export MPICH_MAX_THREAD_SAFETY=multiple
export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0
export SLURM_CPU_BIND="none"

export TF_CPP_MIN_LOG_LEVEL=2

# cd /pscratch/sd/f/fernando/h1_asymmetry

# echo python hvd_unfolding.py Rapgap closure 0
# srun python hvd_unfolding.py Rapgap closure 0

echo python train_hvd.py ../config/config.yaml
srun python train_hvd.py ../config/config.yaml
