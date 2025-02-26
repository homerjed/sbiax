#!/bin/bash
#SBATCH --job-name=arch
#SBATCH --output=%A/nle_arch_search_%A_%a.out
#SBATCH --error=%A/nle_arch_search_%A_%a.err
#SBATCH --array=0-1%2
#SBATCH --time=48:00:00
#SBATCH --partition=cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20 ## This number divided by n_processes is number of parallel runs
#SBATCH --mem=16G
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail

# Run multiple hyperparameter search jobs, each multiprocessing individual search jobs.

id=$SLURM_ARRAY_TASK_ID

echo "Currently running array index: " "${id}"

# Activate env
source /project/ls-gruen/users/jed.homer/sbipdf/.venv/bin/activate

# Run job, ensuring same log file with --exp_name
cd /project/ls-gruen/users/jed.homer/sbipdf/scripts/sbi/moments/

# # Be sure to re-init the exp name, job halts otherwise

# # echo "Running NLE CNF on id: " "${id}"
# # JAX_PLATFORMS=cpu python arch_search.py \
# #     --exp_name "_nle_cnf" --model_type "CNF" --multiprocess --n_processes 20 --n_parallel 1 # 2 CPUs per trial being run

echo "Running NLE MAF on id: " "${id}"
python arch_search.py --exp_name "nle_maf" --model_type "maf" --multiprocess --n_processes 10 --n_parallel 1 # 2 CPUs per trial being run

## SBATCH --output=/project/ls-gruen/users/jed.homer/sbifpdf/scripts/sbi/moments/sbatch_outs/optuna/%A/nle_arch_search_%A_%a.out
## SBATCH --error=/project/ls-gruen/users/jed.homer/sbifpdf/scripts/sbi/moments/sbatch_outs/optuna/%A/nle_arch_search_%A_%a.err