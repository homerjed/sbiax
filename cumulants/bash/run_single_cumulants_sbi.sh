#!/bin/bash
#SBATCH --job-name=sbi_single
#SBATCH --output=sbatch_out/single_sbi.out
#SBATCH --error=sbatch_out/single_sbi.err
#SBATCH --partition=cluster
#SBATCH --time=00:02:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

mkdir -p "logs/"

export LOG_DIR="logs/"
export LOG_LEVEL="DEBUG"
export RESULTS_DIR="results/"
export DEFAULT_NDE_TYPE="CNF"
export DEFAULT_N_NDES="1"
export FORCE_NOISELESS_DATAVECTOR="False"
export USE_QUIJOTE_TAILS="False"
export FIDUCIAL_REDUCE="True"
export DEFAULT_RESOLUTION="1024"
export NON_GAUSSIAN_TEST="False"

SEED=6
COMPRESSION="linear" # LINEAR
LINEARISED="--linearised" # LINEARISED
PRETRAIN="--no-pre-train"
N_LINEAR_SIMS=2000

REDSHIFT="0.0"
SCALES="5.0 10.0 15.0 20.0 25.0 30.0 35.0"
ORDER_IDX="0 1 2" 

python cumulants_sbi.py \
--seed $SEED \
--compression $COMPRESSION \
$LINEARISED \
$PRETRAIN \
--n_linear_sims $N_LINEAR_SIMS \
--order_idx $ORDER_IDX \
--scales $SCALES \
--redshift $REDSHIFT \
--use-tqdm \
--bulk_or_tails "bulk" \
--no-use-planck \
--no-freeze-parameters

python cumulants_sbi.py \
--seed $SEED \
--compression $COMPRESSION \
$LINEARISED \
$PRETRAIN \
--n_linear_sims $N_LINEAR_SIMS \
--order_idx $ORDER_IDX \
--scales $SCALES \
--redshift $REDSHIFT \
--use-tqdm \
--bulk_or_tails "tails" \
--no-use-planck \
--no-freeze-parameters