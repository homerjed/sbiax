#!/bin/bash
#SBATCH --job-name=arch
#SBATCH --output=/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/%A/arch_search_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/%A/nle_arch_search_%A_%a.err
#SBATCH --array=0%1
#SBATCH --time=48:00:00
#SBATCH --partition=cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20 ## This number divided by n_processes is number of parallel runs
#SBATCH --mem=16G
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail

# Run multiple hyperparameter search jobs, each multiprocessing individual search jobs.

id=$SLURM_ARRAY_TASK_ID

echo ">>Currently running array index: " "${id}"

# Activate env
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

# Run job, ensuring same log file with --exp_name
cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/

# Be sure to re-init the exp name, job halts otherwise (--multiprocess --n_processes 10 --n_parallel 1 # 2 CPUs per trial being run)
echo ">>Running arch search on id: " "${id}"
echo ">>order_idx 0 1 2"
python arch_search.py --seed 0 --redshift 0.0 --order_idx 0 1 2 --no-linearised --no-reduced_cumulants
# echo ">>order_idx 0 1"
# python arch_search.py --seed 0 --redshift 0.0 --order_idx 0 --no-linearised  
# echo ">>order_idx 0 1"
# python arch_search.py --seed 0 --redshift 0.0 --order_idx 0 1 --no-linearised  