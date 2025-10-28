#!/bin/bash
# -------------------------------------------------------------
# Submit one SLURM job per redshift to run test_dataset.py
# Each job handles both bulk & tails sequentially.
# -------------------------------------------------------------

# ==== user-configurable flags (applied to ALL jobs) ====
DELTAS_CUT="False"
PER_PDF_CDF_CUT="False"

# config
REDSHIFTS=(0.0 0.5 1.0)
SBATCH_TIME="00:45:00"
SBATCH_MEM="32GB"
SBATCH_CPUS=16
SBATCH_PARTITION="cluster"
SBATCH_MAIL="jed.homer@physik.lmu.de"

BASE_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants"
VENV_PATH="/project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate"
OUT_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/datasets"

mkdir -p "$OUT_DIR"

# experiment settings
USE_SOBOL="True"
if [ "$USE_SOBOL" == "True" ]; then
    SCALES="5.9 9.8 13.7 17.6 21.5 25.4 29.3 33.2"
    N_LINEAR_SIMS=32768
else
    SCALES="5.0 10.0 15.0 20.0 25.0 30.0 35.0"
    N_LINEAR_SIMS=2000
fi

SEED=0
COMPRESSION="nn"
ORDER_IDX="0 1 2"

for Z in "${REDSHIFTS[@]}"; do
  JOB_FILE=$(mktemp)

  cat > "$JOB_FILE" <<END
#!/bin/bash
#SBATCH --job-name=dataset_z${Z}
#SBATCH --output=${OUT_DIR}/z${Z}.out
#SBATCH --error=${OUT_DIR}/z${Z}.err
#SBATCH --partition=${SBATCH_PARTITION}
#SBATCH --time=${SBATCH_TIME}
#SBATCH --mem=${SBATCH_MEM}
#SBATCH --cpus-per-task=${SBATCH_CPUS}
#SBATCH --mail-user=${SBATCH_MAIL}
#SBATCH --mail-type=begin,end,fail

set -euo pipefail

cd ${BASE_DIR}
source ${VENV_PATH}

# ---------- Flags (from parent submit script) ----------
export DELTAS_CUT="${DELTAS_CUT}"
export PER_PDF_CDF_CUT="${PER_PDF_CDF_CUT}"
export USE_SOBOL="${USE_SOBOL}"

# ---------- Build a single suffix for subdir ----------
suffix=""
if [ "\$DELTAS_CUT" = "True" ]; then
  suffix="deltas"
fi
if [ "\$PER_PDF_CDF_CUT" = "True" ]; then
  if [ -n "\$suffix" ]; then
    suffix="\${suffix}_PERPDFCDF"
  else
    suffix="PERPDFCDF"
  fi
fi
# If neither flag is set, use a fallback label
if [ -z "\$suffix" ]; then
  suffix="fiducial_cut"
fi

# ---------- Final results + logs directories ----------
RESULTS_DIR_BASE="results/run_datasets"
LOGS_DIR_BASE="logs/run_datasets"

export RESULTS_DIR="\${RESULTS_DIR_BASE}/\${suffix}/"
export LOG_DIR="\${LOGS_DIR_BASE}/\${suffix}/"
mkdir -p "\$RESULTS_DIR" "\$LOG_DIR"

# ---------- Other env ----------
export LOG_LEVEL="DEBUG"
export PRINT_LOGS="True"
export FORCE_NOISELESS_DATAVECTOR="False"
export FORCE_RECOMPUTE_DATASET="True"
export USE_QUIJOTE_TAILS="False"
export FIDUCIAL_REDUCE="False"
export DEFAULT_RESOLUTION="1024"
export NON_GAUSSIAN_TEST="False"
export USE_PARALLEL_DATALOADING="True"
export N_JOBS="\${SLURM_CPUS_PER_TASK}"
export DATASET_TEST="False"

# ---------- Experiment hyperparams ----------
SEED=${SEED}
COMPRESSION="${COMPRESSION}"
ORDER_IDX="${ORDER_IDX}"
SCALES="${SCALES}"
N_LINEAR_SIMS=${N_LINEAR_SIMS}
REDSHIFT=${Z}

run_once () {
  local FLAG="\$1"
  echo "[z=\${REDSHIFT}] Running \${FLAG}  -> RESULTS_DIR=\$RESULTS_DIR  LOG_DIR=\$LOG_DIR"
  uv run python test_dataset.py \\
    --seed "\${SEED}" \\
    --compression "\${COMPRESSION}" \\
    --no-linearised \\
    --no-pre-train \\
    --n_linear_sims "\${N_LINEAR_SIMS}" \\
    --order_idx \${ORDER_IDX} \\
    --scales \${SCALES} \\
    --redshift "\${REDSHIFT}" \\
    --use-tqdm \\
    --bulk_or_tails "\${FLAG}"
}

for FLAG in bulk tails; do
  run_once "\${FLAG}"
done
END

  JOB_ID=$(sbatch "$JOB_FILE" | awk '{print $4}')
  echo "Submitted redshift $Z as job $JOB_ID"
  rm "$JOB_FILE"
done
