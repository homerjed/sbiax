#!/bin/bash

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Run `N_SEEDS_GLOBAL` repeated cumulants-SBI experiments for different redshifts and scales
# where `N_SEEDS` independent datavectors are drawn to sample posteriors with 
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

get_log_dir() {
    local results_dir="$1"
    shift  # Remove first argument
    local joined=""
    for part in "$@"; do
        joined="${joined}/${part}"
    done
    echo "${results_dir%/}$joined"
}

# Command line args
RESULTS_DIR="results/${1:-"results/"}"
SINGLE_RUN="${2:-false}" # Run all experiments once for a single figure one
ONLY_RUN_FIGURES="${3:-false}" # Only run figure_one.py jobs

RUN_LINEARISED=true
RUN_NONLINEAR=true

RUN_FROZEN=false
USE_PLANCK=false

DEFAULT_NDE_TYPE="CNF"
DEFAULT_N_NDES=1 
USE_SCALERS=true

FORCE_FLAT_PRIOR=false
FORCE_QUIJOTE_PRIOR=true
NON_GAUSSIAN_TEST=false
PLOT_FISHER_CLIPPED=true
FORCE_RECOMPUTE_DATASET=false

COMPRESSION="ensemble-nn"
N_ENSEMBLE_NETS=10
NN_TYPE="NN"

USE_PRECISION_NN=false
DATA_PROCESS_TYPE_NN="dp"

USE_SOBOL=true
FORCE_NOISELESS_DATAVECTOR=false
USE_QUIJOTE_TAILS=false # Use tails datavectors measured, not calculated, from Quijote
FIDUCIAL_REDUCE=true # Reduce cumulants with fiducial variances
DEFAULT_RESOLUTION=1024

RECALCULATE_DATASETS=false
DATASETS_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/quijote_data/datasets/"

N_DATAVECTORS=1 # Number of independent datavectors to sample posteriors with (with N_SEEDS different posteriors)

# Posterior sampling
AFFINE_SAMPLE=false
BLACKJAX_SAMPLE=true

# Running a test single run or not
if [[ "$SINGLE_RUN" == "true" ]]; then
    echo "SINGLE RUN."
    N_SEEDS=1
    START_SEED=0
    N_SEEDS_GLOBAL=1    # Number of repeated trainings for SBI 
    END_SEED=$(( $START_SEED + $N_SEEDS - 1 ))
    N_PARALLEL=2
else
    echo "MULTIPLE SEEDS RUN."
    N_SEEDS=20 # 100 # Number of independent datavectors to test each SBI with
    START_SEED=0
    N_SEEDS_GLOBAL=4 # 9 # Number of repeated trainings for SBI
    END_SEED=$(( $START_SEED + $N_SEEDS - 1 ))
    N_PARALLEL=100
fi

N_GB=12
N_CPU=8
JOB_TIME="16:00:00"
MAIL_TYPE="begin,end,fail"
JOB_ARRAY_STR="$START_SEED-$END_SEED%$N_PARALLEL"

order_idxs=(
    # "0"
    "0 1 2"
)

if [ "$USE_SOBOL" == "true" ]; then
    # SCALES="5.8 9.7 13.6 17.5 21.4 25.3 29.2 33.2"
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

all_redshifts=(
    0.0 
    0.5 
    1.0
)

# SBATCH out directory
TIMESTAMP=$(date +'%m%d_%H%M')
OUT_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/cumulants_sbi/$TIMESTAMP"
mkdir -p "$OUT_DIR"

# Empty datasets dir, recalculate them only once each
if [ "$RECALCULATE_DATASETS" == true ]; then
    rm -rf "${DATASETS_DIR:?}/"*                      # NOTE: not recalculating datasets here
    echo "Emptied datasets directory: $DATASETS_DIR"
else
    echo "Didn't empty datasets directory: $DATASETS_DIR"
fi

# Base logging directory
BASE_LOG_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/$RESULTS_DIR/logs/"
mkdir -p "$BASE_LOG_DIR"

LOG_LEVEL="DEBUG"
USE_TQDM="--use-tqdm"

if [ "$USE_PLANCK" == true ]; then 
    USE_PLANCK_FLAG="--use-planck"
else
    USE_PLANCK_FLAG="--no-use-planck"
fi

# Run cumulants-from-quijote if using high resolution
data_deps=()
data_dep_string=""

if [[ "$USE_QUIJOTE_TAILS" == "true" && "$USE_SOBOL" == "false" ]]; then
    cumulants_data_log_dir=$(get_log_dir \
        $BASE_LOG_DIR \
        "cumulants_data" 
    )

    cumulants_cmd="\
    #!/bin/bash
    #SBATCH --job-name=cumulants_data
    #SBATCH --output=$OUT_DIR/cumulants_data.out
    #SBATCH --error=$OUT_DIR/cumulants_data.err
    #SBATCH --partition=cluster
    #SBATCH --time=$JOB_TIME
    #SBATCH --mem=${N_GB}GB
    #SBATCH --cpus-per-task=$N_CPU
    #SBATCH --mail-user=jed.homer@physik.lmu.de
    #SBATCH --mail-type=$MAIL_TYPE

    cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/data/
    source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

    mkdir -p "$cumulants_data_log_dir"

    export N_DATAVECTOR_SEEDS=$N_SEEDS 
    export N_REPEATED_SBI_SEEDS=$N_SEEDS_GLOBAL 
    export LOG_DIR="${cumulants_data_log_dir}/"
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
    export USE_SCALERS="$USE_SCALERS"
    export BLACKJAX_SAMPLE="$BLACKJAX_SAMPLE"
    export AFFINE_SAMPLE="$AFFINE_SAMPLE"
    export N_ENSEMBLE_NETS="$N_ENSEMBLE_NETS"
    export NN_TYPE="$NN_TYPE"

    echo \"Running cumulants data script\"

    uv run python get_cumulants_data.py
    "

    echo "Running cumulants data scripts"

    data_job_id=$(echo "$cumulants_cmd" | sbatch | awk '{print $4}')
    data_deps+=("$data_job_id")
else
    echo "Not running cumulants data scripts"
fi

if [[ "$USE_SOBOL" == "false" ]]; then
    # Always run PDFs
    pdfs_data_log_dir=$(get_log_dir \
        $BASE_LOG_DIR \
        "pdfs_data"
    )

    pdfs_cmd="\
    #!/bin/bash
    #SBATCH --job-name=pdfs_data
    #SBATCH --output=$OUT_DIR/pdfs_data.out
    #SBATCH --error=$OUT_DIR/pdfs_data.err
    #SBATCH --partition=cluster
    #SBATCH --time=$JOB_TIME
    #SBATCH --mem=${N_GB}GB
    #SBATCH --cpus-per-task=$N_CPU
    #SBATCH --mail-user=jed.homer@physik.lmu.de
    #SBATCH --mail-type=$MAIL_TYPE

    cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/data/
    source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

    mkdir -p "$pdfs_data_log_dir"

    export N_DATAVECTOR_SEEDS=$N_SEEDS 
    export N_REPEATED_SBI_SEEDS=$N_SEEDS_GLOBAL 
    export LOG_DIR="${pdfs_data_log_dir}/"
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
    export AFFINE_SAMPLE="$AFFINE_SAMPLE"
    export N_ENSEMBLE_NETS="$N_ENSEMBLE_NETS"
    export NN_TYPE="$NN_TYPE"

    echo \"Running PDFs data script\"

    uv run python get_pdf_data.py
    "

    echo "Running PDF data scripts"

    data_job_id=$(echo "$pdfs_cmd" | sbatch | awk '{print $4}')
    data_deps+=("$data_job_id")
    data_dep_string=$(IFS=':'; echo "${data_deps[*]}")
fi

# If not running data jobs, quickly submit an empty job
# to keep the data_dep_string non-empty
if [ ${#data_deps[@]} -eq 0 ]; then
    dummy_job_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=dummy
#SBATCH --time=00:01:00
#SBATCH --mem=1M
true
EOF
    )

    # Initialize dependencies with this dummy job ID
    data_deps=("$dummy_job_id")

    # Always build data_dep_string from the array
    data_dep_string=$(IFS=':'; echo "${data_deps[*]}")
fi

# Run SBI jobs
for global_seed in $(seq 0 $N_SEEDS_GLOBAL); do
for FREEZE_FLAG in "--freeze-parameters" "--no-freeze-parameters"; do
    for LINEARISED_FLAG in "--linearised" "--no-linearised"; do
        for PRETRAIN_FLAG in "--pre-train" "--no-pre-train"; do
            for bt in "bulk" "tails"; do

                # Skip pre-train experiments
                if [[ "$PRETRAIN_FLAG" == "--pre-train" ]]; then
                    continue
                fi

                # Skip linearised if not requested
                if [[ "$RUN_LINEARISED" == false && "$LINEARISED_FLAG" == "--linearised" ]]; then
                    echo "Skipping SBI (linearised)"
                    continue
                fi

                # Skip non-linearised if not requested
                if [[ "$RUN_NONLINEAR" == false && "$LINEARISED_FLAG" == "--no-linearised" ]]; then
                    echo "Skipping SBI (non-linearised)"
                    continue
                fi

                # Skip freezing parameters if not requested
                if [[ "$RUN_FROZEN" == false && "$FREEZE_FLAG" == "--freeze-parameters" ]]; then
                    continue
                fi

                # Skip linearised runs if using Planck prior
                if [[ "$FREEZE_FLAG" == "--freeze-parameters" && "$USE_PLANCK" == true ]]; then
                    continue
                fi

                if [ "$bt" == "bulk" ]; then # Label runs
                    bt_flag="b"
                else
                    bt_flag="t"
                fi

                if [ "$LINEARISED_FLAG" == "--linearised" ]; then # Label runs
                    l_flag="l"
                else
                    l_flag="nl"
                fi

                if [ "$FREEZE_FLAG" == "--freeze-parameters" ]; then # Label runs
                    f_flag="f"
                else
                    f_flag="nf"
                fi

                for scale_args in "${scales_sets[@]}"; do
                for order_idx_args in "${order_idxs[@]}"; do

                    # SBI job IDs that must all run for cumulants_multi_z.py to run for bulk/tails, linearised/no-linearised, freeze/no-freeze
                    sbi_job_ids=()
                    # for z in all_redshifts; do
                    for z in "${all_redshifts[@]}"; do

                        echo ">>Running SBI with z=$z, bulk/tails=$bt, cumulants=$order_idx_args, pretrain=$PRETRAIN_FLAG, linearised=$LINEARISED_FLAG, freeze=$FREEZE_FLAG"

                        order_idx_str=$(echo "$order_idx_args" | tr -d ' ')
                        scale_str=$(echo "$scale_args" | tr -d ' ')

                        sbi_log_dir=$(get_log_dir \
                            $BASE_LOG_DIR \
                            "sbi" \
                            $z \
                            $bt \
                            $order_idx_str \
                            $scale_str \
                            $LINEARISED_FLAG \
                            $PRETRAIN_FLAG \
                            $global_seed
                        )

                        sbi_cmd="\
uv run python cumulants_sbi.py \
--seed $global_seed \
--compression $COMPRESSION \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--n_linear_sims $N_LINEAR_SIMS \
--order_idx "$order_idx_args" \
--scales "$scale_args" \
--redshift $z \
--no-use-tqdm \
--bulk_or_tails $bt \
$USE_PLANCK_FLAG \
$FREEZE_FLAG"

                        sbi_job_script=$(
                            cat <<END
#!/bin/bash
#SBATCH --job-name=sbi_${global_seed}_${bt_flag}_${l_flag}_${f_flag}_z${z}
#SBATCH --output=$OUT_DIR/sbi/${bt_flag}/${l_flag}/${f_flag}/z${z}/sbi_fixed_%j.out
#SBATCH --error=$OUT_DIR/sbi/${bt_flag}/${l_flag}/${f_flag}/z${z}/sbi_fixed_%j.err
#SBATCH --partition=cluster
#SBATCH --time=$JOB_TIME
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --dependency=afterok:$data_dep_string

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

mkdir -p "$sbi_log_dir"

export LOG_DIR="$sbi_log_dir"
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
export AFFINE_SAMPLE="$AFFINE_SAMPLE"
export N_ENSEMBLE_NETS="$N_ENSEMBLE_NETS"
export NN_TYPE="$NN_TYPE"

echo "Running sbi script with seed $global_seed and redshift $z"
$sbi_cmd
END
                        )

                        sbi_job_id=$(echo "$sbi_job_script" | sbatch | awk '{print $4}')

                        sbi_job_ids+=("$sbi_job_id")

                    done

                    sbi_deps=$(
                        IFS=":"
                        echo "${sbi_job_ids[*]}"
                    )

                    # Should place these next to 
                    multi_z_job_ids=()
                    figure_one_ids=() # NOTE: this should be outside of bulk / tails since it needs both to run
                    
                    if [[ "$ONLY_RUN_FIGURES" == false ]]; then

                        order_idx_str=$(echo "$order_idx_args" | tr -d ' ')
                        scale_str=$(echo "$scale_args" | tr -d ' ')
                        redshifts_str="${all_redshifts[*]}"

                        multi_z_log_dir=$(get_log_dir \
                            $BASE_LOG_DIR \
                            "multi_z" \
                            $bt \
                            $order_idx_str \
                            $scale_str \
                            $LINEARISED_FLAG \
                            $PRETRAIN_FLAG \
                            $global_seed
                        )

                        multi_z_cmd="\
uv run python cumulants_multi_z.py \
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

                        # Run multi-z posterior sampling after all redshift SBI experiments are run 
                        # (separately for bulk and tails)
                        multi_z_script=$(
                            cat <<END
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
#SBATCH --dependency=afterok:$data_dep_string:$sbi_deps

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
export USE_PRECISION_NN="$USE_PRECISION_NN"
export USE_SCALERS="$USE_SCALERS"
export BLACKJAX_SAMPLE="$BLACKJAX_SAMPLE"
export AFFINE_SAMPLE="$AFFINE_SAMPLE"
export N_ENSEMBLE_NETS="$N_ENSEMBLE_NETS"
export NN_TYPE="$NN_TYPE"

echo "Running final multi-z script"
$multi_z_cmd
END
                        )

                        echo ">>Running Multi-z with global_seed=$global_seed, datavector_seed=$SLURM_ARRAY_TASK_ID, z=$z, bulk/tails=$bt, cumulants=$order_idx_args, pretrain=$PRETRAIN_FLAG, linearised=$LINEARISED_FLAG, freeze=$FREEZE_FLAG"
                        
                        multi_z_job_id=$(echo "$multi_z_script" | sbatch | awk '{print $4}') # FINAL SCRIPT not JOB SCRIPT (sbi)

                        multi_z_job_ids+=("$multi_z_job_id")

                        multi_z_deps=$(
                            IFS=":"
                            echo "${multi_z_job_ids[*]}"
                        )

                    fi

                    order_idx_str=$(echo "$order_idx_args" | tr -d ' ')
                    scale_str=$(echo "$scale_args" | tr -d ' ')

                    figure_one_log_dir=$(get_log_dir \
                        $BASE_LOG_DIR \
                        "figure_one" \
                        $bt \
                        $order_idx_str \
                        $scale_str \
                        $LINEARISED_FLAG \
                        $PRETRAIN_FLAG \
                        $global_seed
                    )

                    figure_cmd="\
uv run python figure_one.py \
--seed $global_seed \
--seed_datavector \$SLURM_ARRAY_TASK_ID \
--n_datavectors $N_DATAVECTORS \
--n_linear_sims $N_LINEAR_SIMS \
--compression $COMPRESSION \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--order_idx "$order_idx_args" \
--scales "$scale_args" \
--redshifts $redshifts_str \
$USE_PLANCK_FLAG \
$FREEZE_FLAG"

                    # Need to run this after both bulk and tails multi_z samplings
                    figure_job=$(
                        cat <<END
#!/bin/bash
#SBATCH --job-name=figure_1_${global_seed}
#SBATCH --output=$OUT_DIR/figure_1/${global_seed}/${l_flag}/${f_flag}/figure_one_%a_%j.out
#SBATCH --error=$OUT_DIR/figure_1/${global_seed}/${l_flag}/${f_flag}/figure_one_%a_%j.err
#SBATCH --array=$JOB_ARRAY_STR
#SBATCH --partition=cluster
#SBATCH --time=$JOB_TIME
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --dependency=afterok:$data_dep_string:$sbi_deps:$multi_z_deps

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

mkdir -p "${figure_one_log_dir}/${SLURM_ARRAY_TASK_ID}"

export LOG_DIR="${figure_one_log_dir}/${SLURM_ARRAY_TASK_ID}"
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
export AFFINE_SAMPLE="$AFFINE_SAMPLE"
export N_ENSEMBLE_NETS="$N_ENSEMBLE_NETS"
export NN_TYPE="$NN_TYPE"

echo "Running final figure one script"
$figure_cmd
END
                    )

                    echo ">>Running figure one with global_seed=$global_seed, datavector_seed=$SLURM_ARRAY_TASK_ID, cumulants=$order_idx_args, pretrain=$PRETRAIN_FLAG, linearised=$LINEARISED_FLAG, freeze=$FREEZE_FLAG"

                    figure_job_id=$(echo "$figure_job" | sbatch | awk '{print $4}')

                    figure_one_ids+=("$figure_job_id")

                    figure_one_deps=$(
                        IFS=":"
                        echo "${figure_one_ids[*]}"
                    )

                done
                done # scales
            done
        done
    done
done
done

##### FIGURE TWO ##### 

# for FREEZE_FLAG in "--freeze-parameters" "--no-freeze-parameters"; do
# for LINEARISED_FLAG in "--linearised" "--no-linearised"; do
# for PRETRAIN_FLAG in "--pre-train" "--no-pre-train"; do
# for scale_args in "${scales_sets[@]}"; do
# for order_idx_args in "${order_idxs[@]}"; do

#     # Skip pre-train experiments
#     if [[ "$PRETRAIN_FLAG" == "--pre-train" ]]; then
#         continue
#     fi

#     # Skip linearised if not requested
#     if [[ "$RUN_LINEARISED" == false && "$LINEARISED_FLAG" == "--linearised" ]]; then
#         echo "Skipping Figure 2 (linearised)"
#         continue
#     fi

#     # Skip non-linearised if not requested
#     if [[ "$RUN_NONLINEAR" == false && "$LINEARISED_FLAG" == "--no-linearised" ]]; then
#         echo "Skipping Figure 2 (non-linearised)"
#         continue
#     fi

#     # Skip freezing parameters if not requested
#     if [[ "$RUN_FROZEN" == false && "$FREEZE_FLAG" == "--freeze-parameters" ]]; then
#         continue
#     fi

#     # Skip linearised runs if using Planck prior
#     if [[ "$FREEZE_FLAG" == "--freeze-parameters" && "$USE_PLANCK" == true ]]; then
#         continue
#     fi
    
#     # Tag for job name
#     if [ "$LINEARISED_FLAG" == "--linearised" ]; then # Label runs
#         l_flag="l"
#     else
#         l_flag="nl"
#     fi

#     # Tag for job name
#     if [ "$FREEZE_FLAG" == "--freeze-parameters" ]; then # Label runs
#         f_flag="f"
#     else
#         f_flag="nf"
#     fi

#     order_idx_str=$(echo "$order_idx_args" | tr -d ' ')
#     scale_str=$(echo "$scale_args" | tr -d ' ')
 
#     echo ">>Running figure two with cumulants=$order_idx_args, pretrain=$PRETRAIN_FLAG, linearised=$LINEARISED_FLAG, freeze=$FREEZE_FLAG"

#     figure_two_log_dir=$(get_log_dir \
#         $BASE_LOG_DIR \
#         "figure_two" \
#         $order_idx_str \
#         $scale_str
#     )

#     sbatch <<END
# #!/bin/bash
# #SBATCH --job-name=figure_2
# #SBATCH --output=$OUT_DIR/figure_2/${l_flag}/${f_flag}/figure_two_%j.out
# #SBATCH --error=$OUT_DIR/figure_2/${l_flag}/${f_flag}/figure_two_%j.err
# #SBATCH --partition=cluster
# #SBATCH --time=$JOB_TIME
# #SBATCH --mem=${N_GB}GB
# #SBATCH --cpus-per-task=$N_CPU
# #SBATCH --mail-user=jed.homer@physik.lmu.de
# #SBATCH --mail-type=$MAIL_TYPE
# #SBATCH --dependency=afterok:$data_dep_string:$sbi_deps:$multi_z_deps:$figure_one_deps

# cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
# source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

# mkdir -p "$figure_two_log_dir"

# export N_DATAVECTOR_SEEDS=$N_SEEDS 
# export N_REPEATED_SBI_SEEDS=$N_SEEDS_GLOBAL 
# export LOG_DIR="${figure_two_log_dir}/"
# export LOG_LEVEL="$LOG_LEVEL"
# export RESULTS_DIR="$RESULTS_DIR"
# export DEFAULT_NDE_TYPE="$DEFAULT_NDE_TYPE"
# export DEFAULT_N_NDES="$DEFAULT_N_NDES"
# export FORCE_NOISELESS_DATAVECTOR="$FORCE_NOISELESS_DATAVECTOR"
# export USE_QUIJOTE_TAILS="$USE_QUIJOTE_TAILS"
# export FIDUCIAL_REDUCE="$FIDUCIAL_REDUCE"
# export DEFAULT_RESOLUTION="$DEFAULT_RESOLUTION"
# export FORCE_FLAT_PRIOR="$FORCE_FLAT_PRIOR"
# export FORCE_QUIJOTE_PRIOR="$FORCE_QUIJOTE_PRIOR"
# export NON_GAUSSIAN_TEST="$NON_GAUSSIAN_TEST"
# export USE_SOBOL="$USE_SOBOL"
# export PLOT_FISHER_CLIPPED="$PLOT_FISHER_CLIPPED"
# export DATA_PROCESS_TYPE_NN="$DATA_PROCESS_TYPE_NN"
# export USE_PRECISION_NN="$USE_PRECISION_NN"
# export USE_SCALERS="$USE_SCALERS"
# export BLACKJAX_SAMPLE="$BLACKJAX_SAMPLE"
# export AFFINE_SAMPLE="$AFFINE_SAMPLE"
# export N_ENSEMBLE_NETS="$N_ENSEMBLE_NETS"
# export NN_TYPE="$NN_TYPE"

# echo "Running final figure two script"

# uv run python figure_two.py \
# --n_datavectors $N_DATAVECTORS \
# --compression $COMPRESSION \
# --n_linear_sims $N_LINEAR_SIMS \
# --order_idx $order_idx_args \
# --scales $scale_args \
# --redshifts $redshifts_str \
# $LINEARISED_FLAG \
# $PRETRAIN_FLAG \
# $USE_PLANCK_FLAG \
# $FREEZE_FLAG
# END
# done
# done
# done
# done
# done

# # Build colon-separated dependency list for use in: --dependency=afterok:...
# if [[ ${#FIG2_JOB_IDS[@]} -gt 0 ]]; then
#     figure_two_deps="$(IFS=:; echo "${FIG2_JOB_IDS[*]}")"
# else
#     figure_two_deps=""
# fi
# export figure_two_deps
# echo "figure_two_deps=$figure_two_deps"

# --------------------------------------------
# Submit Figure 2 jobs and collect dependencies
# --------------------------------------------
FIG2_JOB_IDS=()

for FREEZE_FLAG in "--freeze-parameters" "--no-freeze-parameters"; do
for LINEARISED_FLAG in "--linearised" "--no-linearised"; do
for PRETRAIN_FLAG in "--pre-train" "--no-pre-train"; do
for scale_args in "${scales_sets[@]}"; do
for order_idx_args in "${order_idxs[@]}"; do

    # Skip pre-train experiments
    if [[ "$PRETRAIN_FLAG" == "--pre-train" ]]; then
        continue
    fi

    # Skip linearised if not requested
    if [[ "$RUN_LINEARISED" == false && "$LINEARISED_FLAG" == "--linearised" ]]; then
        echo "Skipping Figure 2 (linearised)"
        continue
    fi

    # Skip non-linearised if not requested
    if [[ "$RUN_NONLINEAR" == false && "$LINEARISED_FLAG" == "--no-linearised" ]]; then
        echo "Skipping Figure 2 (non-linearised)"
        continue
    fi

    # Skip freezing parameters if not requested
    if [[ "$RUN_FROZEN" == false && "$FREEZE_FLAG" == "--freeze-parameters" ]]; then
        continue
    fi

    # Skip linearised runs if using Planck prior
    if [[ "$FREEZE_FLAG" == "--freeze-parameters" && "$USE_PLANCK" == true ]]; then
        continue
    fi

    # Labels for job name
    if [[ "$LINEARISED_FLAG" == "--linearised" ]]; then
        l_flag="l"
    else
        l_flag="nl"
    fi

    if [[ "$FREEZE_FLAG" == "--freeze-parameters" ]]; then
        f_flag="f"
    else
        f_flag="nf"
    fi

    order_idx_str=$(echo "$order_idx_args" | tr -d ' ')
    scale_str=$(echo "$scale_args" | tr -d ' ')

    echo ">>Submitting Figure 2 with cumulants=$order_idx_args, linearised=$LINEARISED_FLAG, freeze=$FREEZE_FLAG"

    figure_two_log_dir=$(get_log_dir \
        "$BASE_LOG_DIR" \
        "figure_two" \
        "$order_idx_str" \
        "$scale_str" \
    )

    # Submit the Figure 2 job (adjust the python entrypoint/args to your setup)
    sbatch_out=$(
        sbatch <<END
#!/bin/bash
#SBATCH --job-name=fig2_${l_flag}_${f_flag}_${order_idx_str}_${scale_str}
#SBATCH --output=$OUT_DIR/figure_two/fig2_%j.out
#SBATCH --error=$OUT_DIR/figure_two/fig2_%j.err
#SBATCH --partition=cluster
#SBATCH --time=$JOB_TIME
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=$MAIL_TYPE

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

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
export DATA_PROCESS_TYPE_NN="$DATA_PROCESS_TYPE_NN"
export USE_PRECISION_NN="$USE_PRECISION_NN"
export USE_SCALERS="$USE_SCALERS"
export BLACKJAX_SAMPLE="$BLACKJAX_SAMPLE"
export AFFINE_SAMPLE="$AFFINE_SAMPLE"
export N_ENSEMBLE_NETS="$N_ENSEMBLE_NETS"
export NN_TYPE="$NN_TYPE"

echo "Running Figure 2 script"

uv run python figure_two.py \
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
    )

    # Parse job ID like: "Submitted batch job 1234567"
    jid=$(echo "$sbatch_out" | awk '{print $4}')
    if [[ -n "$jid" ]]; then
        FIG2_JOB_IDS+=("$jid")
        echo "   -> Figure 2 job submitted: $jid"
    else
        echo "   !! Failed to capture job ID for this submission"
    fi

done
done
done
done
done

# Build colon-separated dependency list for use in: --dependency=afterok:...
if [[ ${#FIG2_JOB_IDS[@]} -gt 0 ]]; then
    figure_two_deps="$(IFS=:; echo "${FIG2_JOB_IDS[*]}")"
else
    figure_two_deps=""
fi
export figure_two_deps
echo "figure_two_deps=$figure_two_deps"



###### COVERAGES ######

for FREEZE_FLAG in "--freeze-parameters" "--no-freeze-parameters"; do
for PRETRAIN_FLAG in "--pre-train" "--no-pre-train"; do
for scale_args in "${scales_sets[@]}"; do
for order_idx_args in "${order_idxs[@]}"; do

    # Skip pre-train experiments
    if [[ "$PRETRAIN_FLAG" == "--pre-train" ]]; then
        continue
    fi

    # Skip freezing parameters if not requested
    if [[ "$RUN_FROZEN" == false && "$FREEZE_FLAG" == "--freeze-parameters" ]]; then
        continue
    fi

    # Skip linearised runs if using Planck prior
    if [[ "$FREEZE_FLAG" == "--freeze-parameters" && "$USE_PLANCK" == true ]]; then
        continue
    fi
    
    # Tag for job name
    if [ "$FREEZE_FLAG" == "--freeze-parameters" ]; then # Label runs
        f_flag="f"
    else
        f_flag="nf"
    fi

    order_idx_str=$(echo "$order_idx_args" | tr -d ' ')
    scale_str=$(echo "$scale_args" | tr -d ' ')
 
    echo ">>Running COVERAGES with cumulants=$order_idx_args, pretrain=$PRETRAIN_FLAG, linearised=$LINEARISED_FLAG, freeze=$FREEZE_FLAG"

    figure_two_log_dir=$(get_log_dir \
        $BASE_LOG_DIR \
        "figure_two" \
        $order_idx_str \
        $scale_str
    )

    sbatch <<END
#!/bin/bash
#SBATCH --job-name=coverages
#SBATCH --output=$OUT_DIR/coverages/coverages_%j.out
#SBATCH --error=$OUT_DIR/coverages/coverages_%j.err
#SBATCH --partition=cluster
#SBATCH --time=$JOB_TIME
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --dependency=afterok:$figure_two_deps

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate


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
export DATA_PROCESS_TYPE_NN="$DATA_PROCESS_TYPE_NN"
export USE_PRECISION_NN="$USE_PRECISION_NN"
export USE_SCALERS="$USE_SCALERS"
export BLACKJAX_SAMPLE="$BLACKJAX_SAMPLE"
export AFFINE_SAMPLE="$AFFINE_SAMPLE"
export N_ENSEMBLE_NETS="$N_ENSEMBLE_NETS"
export NN_TYPE="$NN_TYPE"

echo "Running final coverages script"

uv run python figure_coverages.py 

END
done
done
done
done