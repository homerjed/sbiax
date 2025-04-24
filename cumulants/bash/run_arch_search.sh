#!/bin/bash

# This script is used to run the arch search for the NLE models.
# - Assume that these hyperparameters work best for all other runs (e.g. tails, bulk, non-linearised, frozen)

TIMESTAMP=$(date +'%m%d_%H%M')
OUT_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/arch_search/$TIMESTAMP"
mkdir -p "$OUT_DIR"

for PRETRAIN_FLAG in "--pre-train" "--no-pre-train"; do
for LINEARISED_FLAG in "--linearised" "--no-linearised"; do

if [[ "$PRETRAIN_FLAG" == "--pre-train" ]]; then
    JOB_NAME="arch_pt"
else
    JOB_NAME="arch_npt"
fi

if [[ "$LINEARISED_FLAG" == "--linearised" ]]; then
    JOB_NAME="${JOB_NAME}_l"
else
    JOB_NAME="${JOB_NAME}_nl"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$OUT_DIR/%A/${JOBNAME}_%A_%a.out
#SBATCH --error=$OUT_DIR/%A/${JOBNAME}_%A_%a.err
#SBATCH --array=0%1
#SBATCH --time=48:00:00
#SBATCH --partition=cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=12G
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail

id=\$SLURM_ARRAY_TASK_ID

echo ">>Currently running array index: \$id"
echo ">>JOB_NAME: $JOB_NAME"
echo ">>Pretrain flag: $PRETRAIN_FLAG"
echo ">>Linearised flag: $LINEARISED_FLAG"

# Activate environment
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/

echo ">>Running arch search on id: \$id"
python arch_search.py --seed 0 --redshift 0.0 --order_idx 0 1 2 --$LINEARISED_FLAG --no-freeze-parameters $PRETRAIN_FLAG

EOF

done
done