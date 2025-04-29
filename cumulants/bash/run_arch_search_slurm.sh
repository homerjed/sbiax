#!/bin/bash

# Submit multiple jobs that each load an optuna study from journal storage 
# and add a trial. 
# - N_JOBS here will run individual trials (or as many as can be run in the time given to the sbatch job)
#   so that 

# --- Config ---
STUDY_NAME="arch_para"
N_JOBS=20
FREEZE_FLAG="--no-freeze-parameters"

# Don't time stamp so storage is in the same place for each slurm job
OUT_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/arch_search/"

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

# # Skip linearised training
# if [[ "$LINEARISED_FLAG" == "--linearised" && "$PRETRAIN_FLAG" == "--pre-train" ]]; then
#     continue
# fi

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

# --- Submit SLURM jobs ---
# MULTI_SLURM environment variable ensures shard journal storage for search across slurm jobs
for i in $(seq 1 $N_JOBS); do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$OUT_DIR/workers/optuna_worker_%j.out
#SBATCH --error=$OUT_DIR/workers/optuna_worker_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

echo ">>JOB_NAME: $JOB_NAME"
echo ">>Pretrain flag: $PRETRAIN_FLAG"
echo ">>Linearised flag: $LINEARISED_FLAG"

# Activate environment
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/

MULTI_SLURM=1 python arch_search_slurm.py \
--seed 0 \
--redshift 0.0 \
--order_idx 0 1 2 \
$LINEARISED_FLAG \
$FREEZE_FLAG \
$PRETRAIN_FLAG

EOF
done
done
done