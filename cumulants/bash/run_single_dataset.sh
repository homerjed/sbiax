#!/bin/bash
#SBATCH --job-name=sbi_single
#SBATCH --output=sbatch_out/single_nn.out
#SBATCH --error=sbatch_out/single_nn.err
#SBATCH --partition=cluster
#SBATCH --time=00:01:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

mkdir -p "logs/"

export LOG_DIR="logs/"
export LOG_LEVEL="DEBUG"
export PRINT_LOGS="True"
export RESULTS_DIR="results/dataset/"
export FORCE_NOISELESS_DATAVECTOR="False"
export FORCE_RECOMPUTE_DATASET="True"
export USE_QUIJOTE_TAILS="False"
export FIDUCIAL_REDUCE="True"
export DEFAULT_RESOLUTION="1024"
export NON_GAUSSIAN_TEST="False"
export USE_SOBOL="True"

export DATASET_TEST="True"

# NOTE: run no-linearised...

SEED=0
COMPRESSION="nn"

REDSHIFT="0.0"
ORDER_IDX="0 1 2" 

if [ "$USE_SOBOL" == "True" ]; then
    # SCALES="5.8 9.7 13.6 17.5 21.4 25.3 29.2 33.2"
    SCALES="5.9 9.8 13.7 17.6 21.5 25.4 29.3 33.2"
    N_LINEAR_SIMS=32768 # This is of order of the sobol sequence
else
    SCALES="5.0 10.0 15.0 20.0 25.0 30.0 35.0" 
    N_LINEAR_SIMS=2000 
fi

for BULK_TAILS_FLAG in "bulk" "tails"; do
    uv run python test_dataset.py \
    --seed $SEED \
    --compression $COMPRESSION \
    --no-linearised \
    --no-pre-train \
    --n_linear_sims $N_LINEAR_SIMS \
    --order_idx $ORDER_IDX \
    --scales $SCALES \
    --redshift $REDSHIFT \
    --use-tqdm \
    --bulk_or_tails $BULK_TAILS_FLAG 
done