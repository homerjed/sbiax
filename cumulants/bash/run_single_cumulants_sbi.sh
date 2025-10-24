#!/bin/bash
#SBATCH --job-name=sbi_single
#SBATCH --output=sbatch_out/single_sbi.out
#SBATCH --error=sbatch_out/single_sbi.err
#SBATCH --partition=cluster
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

mkdir -p "logs/"

export LOG_DIR="logs/"
export LOG_LEVEL="DEBUG"
export RESULTS_DIR="results/maf_test/"

export DEFAULT_NDE_TYPE="CNF"
export DEFAULT_N_NDES=1

export FORCE_RECOMPUTE_DATASET=false
export FORCE_NOISELESS_DATAVECTOR=false

export FORCE_FLAT_PRIOR=false
export FORCE_QUIJOTE_PRIOR=true

export USE_QUIJOTE_TAILS=false
export FIDUCIAL_REDUCE=true # !
export DEFAULT_RESOLUTION=1024

export NON_GAUSSIAN_TEST=false
export USE_SOBOL=true
export PLOT_FISHER_CLIPPED=true

COMPRESSION="linear" # Linear or neural network
export N_ENSEMBLE_NETS=10
export USE_PRECISION_NN=false
export NN_TYPE="NN"
export DATA_PROCESS_TYPE_NN="dp"
export NN_CLIP_NORM=true # Global clipping of weights
export COVARIANCE_NN=true

PRETRAIN="--no-pre-train"

REDSHIFT="0.0"
ORDER_IDX="0 1 2" 

if [ "$USE_SOBOL" == true ]; then
    # SCALES="5.8 9.7 13.6 17.5 21.4 25.3 29.2 33.2"
    SCALES="5.9 9.8 13.7 17.6 21.5 25.4 29.3 33.2"
    N_LINEAR_SIMS=32768 
else
    SCALES="5.0 10.0 15.0 20.0 25.0 30.0 35.0" 
    N_LINEAR_SIMS=2000 
fi

for LINEARISED_FLAG in "--linearised" "--no-linearised"; do
    for SEED in {0..20}; do
        echo "Running SEED=$SEED"

        # if [[ "$LINEARISED_FLAG" == "--linearised" ]]; then
        #     continue
        # fi
        # python cumulants_sbi.py \
        #     --seed $SEED \
        #     --compression $COMPRESSION \
        #     $LINEARISED_FLAG \
        #     $PRETRAIN \
        #     --n_linear_sims $N_LINEAR_SIMS \
        #     --order_idx $ORDER_IDX \
        #     --scales $SCALES \
        #     --redshift $REDSHIFT \
        #     --use-tqdm \
        #     --bulk_or_tails "tails" \
        
        python cumulants_sbi.py \
            --seed $SEED \
            --compression $COMPRESSION \
            $LINEARISED_FLAG \
            $PRETRAIN \
            --n_linear_sims $N_LINEAR_SIMS \
            --order_idx $ORDER_IDX \
            --scales $SCALES \
            --redshift $REDSHIFT \
            --use-tqdm \
            --bulk_or_tails "bulk"  

        python cumulants_sbi.py \
            --seed $SEED \
            --compression $COMPRESSION \
            $LINEARISED_FLAG \
            $PRETRAIN \
            --n_linear_sims $N_LINEAR_SIMS \
            --order_idx $ORDER_IDX \
            --scales $SCALES \
            --redshift $REDSHIFT \
            --use-tqdm \
            --bulk_or_tails "tails" 
        
        # wait  # wait for both background jobs for this seed to finish
    done
done

    # for SEED in {0..10}; do
    #     python cumulants_sbi.py \
    #     --seed $SEED \
    #     --compression $COMPRESSION \
    #     $LINEARISED \
    #     $PRETRAIN \
    #     --n_linear_sims $N_LINEAR_SIMS \
    #     --order_idx $ORDER_IDX \
    #     --scales $SCALES \
    #     --redshift $REDSHIFT \
    #     --use-tqdm \
    #     --bulk_or_tails "bulk" \
    #     --no-use-planck \
    #     --no-freeze-parameters

    #     python cumulants_sbi.py \
    #     --seed $SEED \
    #     --compression $COMPRESSION \
    #     $LINEARISED \
    #     $PRETRAIN \
    #     --n_linear_sims $N_LINEAR_SIMS \
    #     --order_idx $ORDER_IDX \
    #     --scales $SCALES \
    #     --redshift $REDSHIFT \
    #     --use-tqdm \
    #     --bulk_or_tails "tails" \
    #     --no-use-planck \
    #     --no-freeze-parameters
    # done

    # SEED=3

    # python cumulants_sbi.py \
    # --seed $SEED \
    # --compression $COMPRESSION \
    # $LINEARISED \
    # $PRETRAIN \
    # --n_linear_sims $N_LINEAR_SIMS \
    # --order_idx $ORDER_IDX \
    # --scales $SCALES \
    # --redshift $REDSHIFT \
    # --use-tqdm \
    # --bulk_or_tails "bulk" \
    # --no-use-planck \
    # --no-freeze-parameters

    # python cumulants_sbi.py \
    # --seed $SEED \
    # --compression $COMPRESSION \
    # $LINEARISED \
    # $PRETRAIN \
    # --n_linear_sims $N_LINEAR_SIMS \
    # --order_idx $ORDER_IDX \
    # --scales $SCALES \
    # --redshift $REDSHIFT \
    # --use-tqdm \
    # --bulk_or_tails "tails" \
    # --no-use-planck \
    # --no-freeze-parameters