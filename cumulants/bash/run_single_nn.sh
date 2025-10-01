#!/bin/bash
#SBATCH --job-name=sbi_single
#SBATCH --output=sbatch_out/single_nn.out
#SBATCH --error=sbatch_out/single_nn.err
#SBATCH --partition=cluster
#SBATCH --time=08:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail

##SBATCH --gres=gpu:1

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

mkdir -p "logs/"

export LOG_DIR="logs/"
export LOG_LEVEL="DEBUG"
export PRINT_LOGS="true"
# export RESULTS_DIR="results/nn/"
export FORCE_NOISELESS_DATAVECTOR="false"
export FORCE_RECOMPUTE_DATASET="false"
export USE_QUIJOTE_TAILS="false"
export FIDUCIAL_REDUCE="true"
export DEFAULT_RESOLUTION="1024"
export NON_GAUSSIAN_TEST="false"
export USE_SOBOL="true"
export PLOT_FISHER_CLIPPED="false"

SEED=0

USE_PRECISION_NN=false

COMPRESSION="nn"
export USE_PRECISION_NN="$USE_PRECISION_NN"
export DATA_PROCESS_TYPE_NN="d"

PRETRAIN="--no-pre-train"

REDSHIFT="0.0"
ORDER_IDX="0 1 2" 

NN_OR_IMNN="nn"

if [[ "$USE_PRECISION_NN" == "true" ]]; then
    export RESULTS_DIR="results/$NN_OR_IMNN/precision/"
else
    export RESULTS_DIR="results/$NN_OR_IMNN/"
fi


if [ "$USE_SOBOL" == "true" ]; then
    # SCALES="5.8 9.7 13.6 17.5 21.4 25.3 29.2 33.2"
    SCALES="5.9 9.8 13.7 17.6 21.5 25.4 29.3 33.2"
    N_LINEAR_SIMS=32768 # This is of order of the sobol sequence
else
    SCALES="5.0 10.0 15.0 20.0 25.0 30.0 35.0" 
    N_LINEAR_SIMS=2000 
fi

for LINEARISED_FLAG in "--linearised" "--no-linearised"; do

    echo "RUNNING $LINEARISED_FLAG"

    # python nn.py \
    # --seed $SEED \
    # --compression $COMPRESSION \
    # $LINEARISED_FLAG \
    # $PRETRAIN \
    # --n_linear_sims $N_LINEAR_SIMS \
    # --order_idx $ORDER_IDX \
    # --scales $SCALES \
    # --redshift $REDSHIFT \
    # --use-tqdm \
    # --bulk_or_tails "bulk" \
    # --no-use-planck \
    # --no-freeze-parameters \

    python $NN_OR_IMNN.py \
    --seed $SEED \
    --compression $COMPRESSION \
    $LINEARISED_FLAG \
    $PRETRAIN \
    --n_linear_sims $N_LINEAR_SIMS \
    --order_idx $ORDER_IDX \
    --scales $SCALES \
    --redshift $REDSHIFT \
    --use-tqdm \
    --bulk_or_tails "bulk" \
    --no-use-planck \
    --no-freeze-parameters &

    python $NN_OR_IMNN.py \
    --seed $SEED \
    --compression $COMPRESSION \
    $LINEARISED_FLAG \
    $PRETRAIN \
    --n_linear_sims $N_LINEAR_SIMS \
    --order_idx $ORDER_IDX \
    --scales $SCALES \
    --redshift $REDSHIFT \
    --use-tqdm \
    --bulk_or_tails "tails" \
    --no-use-planck \
    --no-freeze-parameters

    wait  # wait for both background jobs for this seed to finish
done