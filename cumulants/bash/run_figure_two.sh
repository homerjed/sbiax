#!/bin/bash

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Re-run figure_two.py for all experimental setups, using existing results
# - NOTE: this uses the same constants from `run_cumulants_sbi.sh`
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

# Command line args
RESULTS_DIR="results/${1:-"results/"}"

echo "RESULTS_DIR: $RESULTS_DIR"

DEFAULT_NDE_TYPE="CNF"
DEFAULT_N_NDES=1 
USE_SCALERS=true # Not implemented yet

DEFAULT_RESOLUTION=1024

COMPRESSION="ensemble-nn"
N_ENSEMBLE_NETS=10
NN_TYPE="NN"
USE_PRECISION_NN=false
DATA_PROCESS_TYPE_NN="dp"

N_DATAVECTORS=1
N_SEEDS=20
N_SEEDS_GLOBAL=1

USE_PLANCK=false
RUN_FROZEN=false
RUN_NONLINEAR=true
USE_SOBOL=true
FORCE_NOISELESS_DATAVECTOR=false
USE_QUIJOTE_TAILS=false # Use tails datavectors measured, not calculated, from Quijote
FIDUCIAL_REDUCE=true # Reduce cumulants with fiducial variances

FORCE_FLAT_PRIOR=false
FORCE_QUIJOTE_PRIOR=true
NON_GAUSSIAN_TEST=false
PLOT_FISHER_CLIPPED=true
FORCE_RECOMPUTE_DATASET=false

# Posterior sampling
BLACKJAX_SAMPLE=true

if [ "$USE_SOBOL" == "true" ]; then
    scales_sets=(
        "5.9 9.8 13.7 17.6 21.5 25.4 29.3 33.2"
    )
    N_LINEAR_SIMS=32768 
else
    scales_sets=(
        "5.0 10.0 15.0 20.0 25.0 30.0 35.0" 
    )
    N_LINEAR_SIMS=2000 
fi

order_idxs=(
    "0 1 2"
)

all_redshifts=(
    0.0 
    0.5 
    1.0
)
redshifts_str="${all_redshifts[*]}"

TIMESTAMP=$(date +'%m%d_%H%M')
OUT_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/rerun_figure_two/$TIMESTAMP"
mkdir -p "$OUT_DIR"

BASE_LOG_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/$RESULTS_DIR/logs/"
mkdir -p "$BASE_LOG_DIR"

LOG_LEVEL="DEBUG"

N_GB=12
N_CPU=8
JOB_TIME="08:00:00"
MAIL_TYPE="begin,end,fail"

# Flags
if [ "$USE_PLANCK" == true ]; then 
    USE_PLANCK_FLAG="--use-planck"
else
    USE_PLANCK_FLAG="--no-use-planck"
fi

for FREEZE_FLAG in "--freeze-parameters" "--no-freeze-parameters"; do
for LINEARISED_FLAG in "--linearised" "--no-linearised"; do
for PRETRAIN_FLAG in "--pre-train" "--no-pre-train"; do
for scale_args in "${scales_sets[@]}"; do
for order_idx_args in "${order_idxs[@]}"; do

    # Skip pre-train experiments
    if [[ "$PRETRAIN_FLAG" == "--pre-train" ]]; then
        continue
    fi

    if [[ "$RUN_NONLINEAR" == false && "$LINEARISED_FLAG" == "--no-linearised" ]]; then
        continue
    fi

    if [[ "$RUN_FROZEN" == false && "$FREEZE_FLAG" == "--freeze-parameters" ]]; then
        continue
    fi

    if [[ "$FREEZE_FLAG" == "--freeze-parameters" && "$USE_PLANCK" == true ]]; then
        continue
    fi
    
    # Tags for naming
    if [ "$LINEARISED_FLAG" == "--linearised" ]; then
        l_flag="l"
    else
        l_flag="nl"
    fi

    if [ "$FREEZE_FLAG" == "--freeze-parameters" ]; then
        f_flag="f"
    else
        f_flag="nf"
    fi

    order_idx_str=$(echo "$order_idx_args" | tr -d ' ')
    scale_str=$(echo "$scale_args" | tr -d ' ')
 
    echo ">>Re-running figure two with cumulants=$order_idx_args, pretrain=$PRETRAIN_FLAG, linearised=$LINEARISED_FLAG, freeze=$FREEZE_FLAG"

    figure_two_log_dir=$(get_log_dir \
        $BASE_LOG_DIR \
        "figure_two_rerun" \
        $order_idx_str \
        $scale_str \
        $l_flag \
        $f_flag
    )

    sbatch <<END
#!/bin/bash
#SBATCH --job-name=fig2_rerun_${l_flag}_${f_flag}
#SBATCH --output=$OUT_DIR/figure_two_${l_flag}_${f_flag}_%j.out
#SBATCH --error=$OUT_DIR/figure_two_${l_flag}_${f_flag}_%j.err
#SBATCH --partition=cluster
#SBATCH --time=$JOB_TIME
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=$MAIL_TYPE

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

mkdir -p "$figure_two_log_dir"

export N_DATAVECTOR_SEEDS=$N_SEEDS 
export N_REPEATED_SBI_SEEDS=$N_SEEDS_GLOBAL 
export LOG_DIR="${figure_two_log_dir}/"
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
export USE_PRECISION_NN="$USE_PRECISION_NN"
export USE_SCALERS="$USE_SCALERS"
export BLACKJAX_SAMPLE="$BLACKJAX_SAMPLE"
export N_ENSEMBLE_NETS="$N_ENSEMBLE_NETS"
export NN_TYPE="$NN_TYPE"

echo "Re-running figure two script"

python figure_two.py \
--n_datavectors $N_DATAVECTORS \
--compression $COMPRESSION \
--n_linear_sims $N_LINEAR_SIMS \
--order_idx $order_idx_args \
--scales $scale_args \
--redshifts $redshifts_str \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
$USE_PLANCK_FLAG \
$FREEZE_FLAG
END

done
done
done
done
done
