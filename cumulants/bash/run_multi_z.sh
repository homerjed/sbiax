#!/bin/bash

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Submit ONLY multi-z posterior sampling jobs (assumes data + per-z SBI already done)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

get_log_dir() {
    local results_dir="$1"
    shift
    local joined=""
    for part in "$@"; do
        joined="${joined}/${part}"
    done
    echo "${results_dir%/}$joined"
}

# -------------------- CLI-like toggles --------------------
RESULTS_DIR="results/${1:-results/}"
SINGLE_RUN="${2:-false}"

RUN_LINEARISED=false
RUN_NONLINEAR=true

RUN_FROZEN=false
USE_PLANCK=false

DEFAULT_NDE_TYPE="CNF"
DEFAULT_N_NDES=1
USE_SCALERS=true

FORCE_FLAT_PRIOR=False
FORCE_QUIJOTE_PRIOR=True
NON_GAUSSIAN_TEST=False
PLOT_FISHER_CLIPPED=True
FORCE_RECOMPUTE_DATASET=False

COMPRESSION="nn"
DATA_PROCESS_TYPE_NN="d"

USE_SOBOL=True
FORCE_NOISELESS_DATAVECTOR=false
USE_QUIJOTE_TAILS=false
FIDUCIAL_REDUCE=true
DEFAULT_RESOLUTION=1024

# -------------------- Job sizing --------------------
if [[ "$SINGLE_RUN" == "true" ]]; then
    echo "SINGLE RUN."
    N_SEEDS=1
    START_SEED=0
    N_SEEDS_GLOBAL=1
    END_SEED=$((START_SEED + N_SEEDS - 1))
    N_PARALLEL=2
else
    echo "MULTIPLE SEEDS RUN."
    N_SEEDS=50
    START_SEED=0
    N_SEEDS_GLOBAL=0
    END_SEED=$((START_SEED + N_SEEDS - 1))
    N_PARALLEL=100
fi

PRETRAIN_FLAG="--no-pre-train"

N_GB=12
N_CPU=8
JOB_TIME="08:00:00"
MAIL_TYPE="begin,end,fail"
JOB_ARRAY_STR="$START_SEED-$END_SEED%$N_PARALLEL"

N_DATAVECTORS=10

order_idxs=(
    "0 1 2"
)

if [ "$USE_SOBOL" == "True" ]; then
    scales_sets=("5.9 9.8 13.7 17.6 21.5 25.4 29.3 33.2")
    N_LINEAR_SIMS=32768
else
    scales_sets=("5.0 10.0 15.0 20.0 25.0 30.0 35.0")
    N_LINEAR_SIMS=2000
fi

all_redshifts=(0.0 0.5 1.0)

# -------------------- Logging / dirs --------------------
TIMESTAMP=$(date +'%m%d_%H%M')
OUT_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/cumulants_multi_z_only/$TIMESTAMP"
mkdir -p "$OUT_DIR"

BASE_LOG_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/$RESULTS_DIR/logs/"
mkdir -p "$BASE_LOG_DIR"

LOG_LEVEL="DEBUG"
USE_TQDM="--use-tqdm"

if [ "$USE_PLANCK" == true ]; then
    USE_PLANCK_FLAG="--use-planck"
else
    USE_PLANCK_FLAG="--no-use-planck"
fi

# -------------------- Submit ONLY multi-z array jobs --------------------
for global_seed in $(seq 0 $N_SEEDS_GLOBAL); do
for FREEZE_FLAG in "--freeze-parameters" "--no-freeze-parameters"; do
    # Skip freezing parameters if not requested
    if [[ "$RUN_FROZEN" == false && "$FREEZE_FLAG" == "--freeze-parameters" ]]; then
        continue
    fi
    # Skip linearised runs if using Planck prior
    if [[ "$FREEZE_FLAG" == "--freeze-parameters" && "$USE_PLANCK" == true ]]; then
        continue
    fi

    for LINEARISED_FLAG in "--linearised" "--no-linearised"; do
        if [[ "$RUN_LINEARISED" == false && "$LINEARISED_FLAG" == "--linearised" ]]; then
            echo "Skipping multi-z (linearised)"
            continue
        fi
        if [[ "$RUN_NONLINEAR" == false && "$LINEARISED_FLAG" == "--no-linearised" ]]; then
            echo "Skipping multi-z (non-linearised)"
            continue
        fi

        # tags
        if [ "$LINEARISED_FLAG" == "--linearised" ]; then l_flag="l"; else l_flag="nl"; fi
        if [ "$FREEZE_FLAG" == "--freeze-parameters" ]; then f_flag="f"; else f_flag="nf"; fi

        for bt in "bulk" "tails"; do
            if [ "$bt" == "bulk" ]; then bt_flag="b"; else bt_flag="t"; fi

            for scale_args in "${scales_sets[@]}"; do
            for order_idx_args in "${order_idxs[@]}"; do

                order_idx_str=$(echo "$order_idx_args" | tr -d ' ')
                scale_str=$(echo "$scale_args" | tr -d ' ')
                redshifts_str="${all_redshifts[*]}"

                multi_z_log_dir=$(get_log_dir \
                    "$BASE_LOG_DIR" \
                    "multi_z_only" \
                    "$bt" \
                    "$order_idx_str" \
                    "$scale_str" \
                    "$LINEARISED_FLAG" \
                    "$global_seed"
                )

                mkdir -p "$OUT_DIR/multi_z/${global_seed}/${bt_flag}/${l_flag}/${f_flag}"
                mkdir -p "$multi_z_log_dir"

                multi_z_cmd="\
python cumulants_multi_z.py \
--seed $global_seed \
--seed_datavector \$SLURM_ARRAY_TASK_ID \
--n_datavectors $N_DATAVECTORS \
--compression $COMPRESSION \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--n_linear_sims $N_LINEAR_SIMS \
--order_idx "$order_idx_args" \
--scales "$scale_args" \
--bulk_or_tails $bt \
--redshifts $redshifts_str \
$USE_PLANCK_FLAG \
$FREEZE_FLAG"

                # NOTE: No data/SBI dependencies; we assume those artifacts already exist.
                sbatch <<END
#!/bin/bash
#SBATCH --job-name=m_z_${global_seed}_${bt_flag}_${l_flag}_${f_flag}
#SBATCH --output=$OUT_DIR/multi_z/${global_seed}/${bt_flag}/${l_flag}/${f_flag}/m_z_%a_%j.out
#SBATCH --error=$OUT_DIR/multi_z/${global_seed}/${bt_flag}/${l_flag}/${f_flag}/m_z_%a_%j.err
#SBATCH --array=$JOB_ARRAY_STR
#SBATCH --partition=cluster
#SBATCH --time=$JOB_TIME
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=$MAIL_TYPE

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

mkdir -p "${multi_z_log_dir}/${SLURM_ARRAY_TASK_ID}"

export LOG_DIR="${multi_z_log_dir}/${SLURM_ARRAY_TASK_ID}"
export LOG_LEVEL="$LOG_LEVEL"
export RESULTS_DIR="$RESULTS_DIR"
export DEFAULT_NDE_TYPE="$DEFAULT_NDE_TYPE"
export DEFAULT_N_NDES="$DEFAULT_N_NDES"
export FORCE_NOISELESS_DATAVECTOR="$FORCE_NOISELESS_DATAVECTOR"
export USE_QUIJOTE_TAILS="$USE_QUIJOTE_TAILS"
export FIDUCIAL_REDUCE="$FIDUCIAL_REDUCE"
export DEFAULT_RESOLUTION="$DEFAULT_RESOLUTION"
export FORCE_FLAT_PRIOR="$FORCE_FLAT_PRIOR"
export FORCE_QUIJOTE_PRIOR="$FORCE_QUIJOTE_PRIOR"
export NON_GAUSSIAN_TEST="$NON_GAUSSIAN_TEST"
export USE_SOBOL="$USE_SOBOL"
export PLOT_FISHER_CLIPPED="$PLOT_FISHER_CLIPPED"
export FORCE_RECOMPUTE_DATASET="$FORCE_RECOMPUTE_DATASET"
export DATA_PROCESS_TYPE_NN="$DATA_PROCESS_TYPE_NN"

echo "Running multi-z only"
$multi_z_cmd
END

            done
            done
        done
    done
done
done
