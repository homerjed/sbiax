#!/bin/bash
#SBATCH --job-name=benchmark_threads
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --hint=nomultithread

# Set threading env variables
# export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
# export MKL_NUM_THREADS=$SLURM_CPUS_ON_NODE
# export OPENBLAS_NUM_THREADS=$SLURM_CPUS_ON_NODE
# export NUMEXPR_NUM_THREADS=$SLURM_CPUS_ON_NODE
# export XLA_FLAGS=--xla_cpu_multi_thread_eigen=true

# Activate virtual environment
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

# Run benchmark
python slurm_speed_test.py
