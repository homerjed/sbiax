#!/bin/bash
#SBATCH --job-name=m_z_test
#SBATCH --output=$OUT_DIR/m_z_test_%a_%j.out
#SBATCH --error=$OUT_DIR/m_z_test_%a_%j.err
#SBATCH --partition=cluster
#SBATCH --time=04:00:00
#SBATCH --mem=12GB
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

# mkdir -p "${multi_z_log_dir}/${SLURM_ARRAY_TASK_ID}"

export LOG_DIR="logs/"
export LOG_LEVEL="DEBUG"
export RESULTS_DIR="results/results/sobol/"

export DEFAULT_NDE_TYPE="CNF"
export DEFAULT_N_NDES="1"

export FORCE_RECOMPUTE_DATASET="False"
export FORCE_NOISELESS_DATAVECTOR="False"

export FORCE_FLAT_PRIOR="False"
export FORCE_QUIJOTE_PRIOR="True"

export USE_QUIJOTE_TAILS="False"
export FIDUCIAL_REDUCE="True" # !
export DEFAULT_RESOLUTION="1024"

export NON_GAUSSIAN_TEST="False"
export USE_SOBOL="True"
export PLOT_FISHER_CLIPPED="True"

COMPRESSION="nn" # Linear or neural network
PRETRAIN="--no-pre-train"

REDSHIFT="0.0"
ORDER_IDX="0 1 2" 

SCALES="5.9 9.8 13.7 17.6 21.5 25.4 29.3 33.2"
N_LINEAR_SIMS=32768 

all_redshifts=(
    0.0 
    0.5 
    1.0
)
redshifts_str="${all_redshifts[*]}"

echo "Running final multi-z script"

python cumulants_multi_z.py \
--seed 0 \
--seed_datavector 0 \
--n_datavectors 10 \
--compression "nn" \
--linearised \
--no-pre-train \
--n_linear_sims $N_LINEAR_SIMS \
--order_idx $ORDER_IDX \
--scales $SCALES \
--bulk_or_tails "bulk" \
--redshifts $redshifts_str \
--no-use-planck \
--no-freeze-parameters