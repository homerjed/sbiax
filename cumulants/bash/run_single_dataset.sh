#!/bin/bash
#SBATCH --job-name=sbi_single
#SBATCH --output=sbatch_out/single_nn.out
#SBATCH --error=sbatch_out/single_nn.err
#SBATCH --partition=cluster
#SBATCH --time=02:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

mkdir -p "logs/"

# export LOG_DIR="logs/"
export LOG_LEVEL="DEBUG"
export PRINT_LOGS="True"
export FORCE_NOISELESS_DATAVECTOR="False"
export USE_QUIJOTE_TAILS="False"
export FIDUCIAL_REDUCE="False"
export DEFAULT_RESOLUTION="1024"
export NON_GAUSSIAN_TEST="False"
export USE_SOBOL="True"

export USE_PARALLEL_DATALOADING="True"
export N_JOBS=16 #$SLURM_CPUS_PER_TASK

export FORCE_RECOMPUTE_DATASET="True"
export DATASET_TEST="True"
export DELTAS_CUT="False"
export PER_PDF_CDF_CUT="False"

RESULTS_DIR="results/dataset"
LOG_DIR="logs/"

suffix=""

if [ "$DELTAS_CUT" = "True" ]; then
  suffix="${suffix}_deltas"
fi

if [ "$PER_PDF_CDF_CUT" = "True" ]; then
  suffix="${suffix}_perpdfcdf"
fi

# If neither flag was set, add the fallback
if [ -z "$suffix" ]; then
  suffix="_fiducial_cut"
fi

# Apply the suffix
RESULTS_DIR="${RESULTS_DIR}${suffix}"
LOG_DIR="${LOG_DIR}${suffix}"

# Add trailing slash
export RESULTS_DIR="${RESULTS_DIR}/"
export LOG_DIR="${LOG_DIR}/"

# NOTE: run no-linearised...

SEED=0
COMPRESSION="nn"

ORDER_IDX="0 1 2" 

if [ "$USE_SOBOL" == "True" ]; then
    SCALES="5.9 9.8 13.7 17.6 21.5 25.4 29.3 33.2"
    N_LINEAR_SIMS=32768 # This is of order of the sobol sequence
else
    SCALES="5.0 10.0 15.0 20.0 25.0 30.0 35.0" 
    N_LINEAR_SIMS=2000 
fi

# list of redshifts to run
REDSHIFTS=(0.0) # 0.5 1.0)

# scales + N_LINEAR_SIMS (Sobol vs non-Sobol)
if [ "$USE_SOBOL" == "True" ]; then
    SCALES="5.9 9.8 13.7 17.6 21.5 25.4 29.3 33.2"
    N_LINEAR_SIMS=32768
else
    SCALES="5.0 10.0 15.0 20.0 25.0 30.0 35.0"
    N_LINEAR_SIMS=2000
fi

PLOT_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/pdf_plots"

if [ -d "$PLOT_DIR" ]; then
  echo "Clearing existing contents of $PLOT_DIR..."
  find "$PLOT_DIR" -mindepth 1 -delete
fi

mkdir -p "$PLOT_DIR"

run_once () {
  local Z="$1"
  local FLAG="$2"
  uv run python test_dataset.py \
    --seed "$SEED" \
    --compression "$COMPRESSION" \
    --no-linearised \
    --no-pre-train \
    --n_linear_sims "$N_LINEAR_SIMS" \
    --order_idx $ORDER_IDX \
    --scales $SCALES \
    --redshift "$Z" \
    --use-tqdm \
    --bulk_or_tails "$FLAG"
}

# loop over redshifts and bulk/tails
for Z in "${REDSHIFTS[@]}"; do
  for BULK_TAILS_FLAG in bulk tails; do # bulk tails
    echo ">>> Running for redshift=$Z, flag=$BULK_TAILS_FLAG"
    run_once "$Z" "$BULK_TAILS_FLAG"
  done
done
