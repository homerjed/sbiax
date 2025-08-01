#!/bin/bash

# Submit multiple jobs that each load an optuna study from journal storage 
# and add a trial. 
# - N_JOBS here will run individual trials (or as many as can be run in the time given to the sbatch job)
#   so that 

# --- Config ---
RESULTS_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/results/" # Base save directory for all results

STUDY_NAME="arch_para"
N_JOBS=10
N_GB=8
N_CPU=8
N_LINEAR_SIMS=10_000 
PARTITION="inter"
LINEAR_ONLY=true # Test NDE or NN on large linearised independent test set
FREEZE_FLAG="--no-freeze-parameters"
NDE_TYPE="MAF"
USE_PLANCK=false

COMPRESSION="nn"

# If requested compression is with a NN, turn arch search onto NN hyperparameters
if [ "$COMPRESSION" == "nn" ]; then
    TEST_COMPRESSION_NN="True"
else
    TEST_COMPRESSION_NN="False"
fi

TIMESTAMP=$(date +'%m%d_%H%M')

# Don't time stamp so storage is in the same place for each slurm job
OUT_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/arch_search/"
mkdir -p "$OUT_DIR"

# Empty the directory where .out/.err are stored for arch search
WORKERS_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/arch_search/workers/"
if [ -d "$WORKERS_DIR" ]; then
    echo "Emptying directory: $WORKERS_DIR"
    rm -rf "${WORKERS_DIR:?}/"*
else
    echo "Directory does not exist: $WORKERS_DIR"
fi

for PRETRAIN_FLAG in "--pre-train" "--no-pre-train"; do
for LINEARISED_FLAG in "--linearised" "--no-linearised"; do

# Skip linearised and pretraining together
if [[ "$LINEARISED_FLAG" == "--linearised" && "$PRETRAIN_FLAG" == "--pre-train" ]]; then
    continue
fi

# Skip linearised training (assume non-linear hyperparameters do well here too)
if [[ "$LINEARISED_FLAG" == "--linearised" && "$PRETRAIN_FLAG" == "--pre-train" ]]; then
    continue
fi
# Skip non-linearised training, using huge independent test set
if [[ "$LINEARISED_FLAG" == "--no-linearised" ]]; then
    continue
fi

# Set flag job names based on args
if [[ "$PRETRAIN_FLAG" == "--pre-train" ]]; then
    JOB_NAME="arch_pt"
else
    JOB_NAME="arch_npt"
fi

# Set flag job names based on args
if [[ "$LINEARISED_FLAG" == "--linearised" ]]; then
    JOB_NAME="${JOB_NAME}_l"
else
    JOB_NAME="${JOB_NAME}_nl"
fi

if [ "$USE_PLANCK" == true ]; then 
    USE_PLANCK_FLAG="--use-planck"
else
    USE_PLANCK_FLAG="--no-use-planck"
fi

# --- Submit SLURM jobs ---
# MULTI_SLURM environment variable ensures shared journal storage for search across slurm jobs
for i in $(seq 1 $N_JOBS); do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$OUT_DIR/$TIMESTAMP/workers/optuna_worker_%j.out
#SBATCH --error=$OUT_DIR/$TIMESTAMP/workers/optuna_worker_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=$PARTITION
#SBATCH --mem=${N_GB}G
#SBATCH --cpus-per-task=${N_CPU}

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

echo ">>JOB_NAME: $JOB_NAME"
echo ">>Pretrain flag: $PRETRAIN_FLAG"
echo ">>Linearised flag: $LINEARISED_FLAG"

# Activate environment
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/

export MULTI_SLURM=1 
export RESULTS_DIR=$RESULTS_DIR 
export DEFAULT_NDE_TYPE=$NDE_TYPE 
export DEFAULT_NDE_TYPE=$NDE_TYPE 
export DEFAULT_N_NDES=1
export TEST_COMPRESSION_NN=$TEST_COMPRESSION_NN

python arch_search_slurm.py \
--seed 0 \
--redshift 0.0 \
--order_idx 0 1 2 \
--n_linear_sims $N_LINEAR_SIMS \
$LINEARISED_FLAG \
$FREEZE_FLAG \
$USE_PLANCK_FLAG \
$PRETRAIN_FLAG

EOF
done
done
done