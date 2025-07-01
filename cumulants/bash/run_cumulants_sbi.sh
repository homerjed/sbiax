#!/bin/bash
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Run `N_SEEDS_GLOBAL` repeated cumulants-SBI experiments for different redshifts and scales
# where `N_SEEDS` independent datavectors are drawn to sample posteriors with
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

get_log_dir() {
    local results_dir="$1"
    shift # Remove first argument
    local joined=""
    for part in "$@"; do
        joined="${joined}/${part}"
    done
    echo "${results_dir%/}$joined"
}

# Command line args
RESULTS_DIR="${1:-"results/"}"
SINGLE_RUN="${2:-false}" # Run all experiments once for a single figure one
ONLY_RUN_FIGURES="${3:-false}" # Only run figure_one.py jobs

RUN_LINEARISED=true
RUN_FROZEN=false
RUN_NONLINEAR=false
USE_PLANCK=false
DEFAULT_NDE_TYPE="CNF"
DEFAULT_N_NDES=1
FORCE_NOISELESS_DATAVECTOR=false
USE_QUIJOTE_TAILS=true # Use tails datavectors measured, not calculated, from Quijote
USE_SCALERS=true # Not implemented yet
N_DATAVECTORS=10 # Number of independent datavectors to sample posteriors with
N_LINEAR_SIMS=2000 # Number of linear simulations to use for training / pre-training
FIDUCIAL_REDUCE=true

order_idxs=(
    # "0"
    "0 1 2"
)
scales_sets=(
    # "15.0 20.0 25.0 30.0 35.0"
    "5.0 10.0 15.0 20.0 25.0 30.0 35.0"
)

# Running a test single run or not
if [[ "$SINGLE_RUN" == "true" ]]; then
    echo "SINGLE RUN."
    N_SEEDS=1
    START_SEED=0
    N_SEEDS_GLOBAL=1 # Number of repeated trainings for SBI
    END_SEED=$(( $START_SEED + $N_SEEDS - 1 ))
    N_PARALLEL=2
    RUN_FROZEN=false
else
    echo "MULTIPLE SEEDS RUN."
    N_SEEDS=100
    START_SEED=0
    N_SEEDS_GLOBAL=10 # Number of repeated trainings for SBI
    END_SEED=$(( $START_SEED + $N_SEEDS ))
    N_PARALLEL=50
fi

N_GB=8
N_CPU=8
JOB_TIME="02:00:00"
MAIL_TYPE="begin,end,fail"
JOB_ARRAY_STR="$START_SEED-$END_SEED%$N_PARALLEL"

# SBATCH out directory
TIMESTAMP=$(date +'%m%d_%H%M')
OUT_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/cumulants_sbi/$TIMESTAMP"
mkdir -p "$OUT_DIR"

# Empty datasets dir, recalculate them only once each
DATASETS_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/quijote_data/datasets/"
rm -rf "${DATASETS_DIR:?}/"*
echo "Emptied datasets directory: $DATASETS_DIR"

# Base logging directory
BASE_LOG_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/$RESULTS_DIR/logs/"
mkdir -p "$BASE_LOG_DIR"
LOG_LEVEL="DEBUG"

if [ "$USE_PLANCK" == true ]; then
    USE_PLANCK_FLAG="--use-planck"
else
    USE_PLANCK_FLAG="--no-use-planck"
fi

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# SBI and Multi-z
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

for global_seed in $(seq 0 $N_SEEDS_GLOBAL); do
    for FREEZE_FLAG in "--freeze-parameters" "--no-freeze-parameters"; do
        for LINEARISED_FLAG in "--linearised" "--no-linearised"; do
            for PRETRAIN_FLAG in "--pre-train" "--no-pre-train"; do

                # Skip pre-train experiments
                if [[ "$PRETRAIN_FLAG" == "--pre-train" ]]; then
                    continue
                fi

                # Skip non-linearised if not requested
                if [[ "$RUN_NONLINEAR" == false && "$LINEARISED_FLAG" == "--no-linearised" ]]; then
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

                for bt in "bulk" "tails"; do

                    if [ "$bt" == "bulk" ]; then
                        # Label runs
                        bt_flag="b"
                    else
                        bt_flag="t"
                    fi

                    if [ "$LINEARISED_FLAG" == "--linearised" ]; then
                        # Label runs
                        l_flag="l"
                    else
                        l_flag="nl"
                    fi

                    if [ "$FREEZE_FLAG" == "--freeze-parameters" ]; then
                        # Label runs
                        f_flag="f"
                    else
                        f_flag="nf"
                    fi

                    for scale_args in "${scales_sets[@]}"; do
                        for order_idx_args in "${order_idxs[@]}"; do

                            # SBI job IDs that must all run for cumulants_multi_z.py to run for bulk/tails, linearised/no-linearised, freeze/no-freeze
                            sbi_job_ids=()

                            for z in 0.0 0.5 1.0; do
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
python cumulants_sbi.py \
--seed $global_seed \
--compression linear \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--n_linear_sims $N_LINEAR_SIMS \
--order_idx \"$order_idx_args\" \
--scales \"$scale_args\" \
--redshift $z \
--no-use-tqdm \
--bulk_or_tails $bt \
$USE_PLANCK_FLAG \
$FREEZE_FLAG"

                                sbi_job_script=$( cat <<EOF
#!/bin/bash
#SBATCH --job-name=sbi_${z}_${bt_flag}_${l_flag}_${f_flag}
#SBATCH --time=$JOB_TIME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mem=${N_GB}GB
#SBATCH --output=$OUT_DIR/sbi_${z}_${bt_flag}_${l_flag}_${f_flag}_%A.out
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --mail-user=jed.homer@gmail.com

source /project/ls-gruen/users/jed.homer/sbi_installs/sbi_venv/bin/activate
cd /project/ls-gruen/users/jed.homer/sbiaxpdf/
mkdir -p $sbi_log_dir
echo "Running SBI with z=$z, bulk/tails=$bt, cumulants=$order_idx_args, pretrain=$PRETRAIN_FLAG, linearised=$LINEARISED_FLAG, freeze=$FREEZE_FLAG"
$sbi_cmd --log_dir $sbi_log_dir
EOF
)
                                # If only running figures, don't run SBI jobs
                                if [ "$ONLY_RUN_FIGURES" == false ]; then
                                    sbi_job_id=$(echo "$sbi_job_script" | sbatch | awk '{print $4}')
                                    sbi_job_ids+=("$sbi_job_id")
                                fi

                            done # z

                            sbi_deps=$( IFS=":" echo "${sbi_job_ids[*]}" )

                            # Run multi-z sampling after all single-z SBI runs are done
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
python cumulants_multi_z.py \
--seed $global_seed \
--seed_datavector \$SLURM_ARRAY_TASK_ID \
--n_datavectors $N_DATAVECTORS \
--n_linear_sims $N_LINEAR_SIMS \
--compression linear \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--order_idx \"$order_idx_args\" \
--scales \"$scale_args\" \
--bulk_or_tails $bt \
$USE_PLANCK_FLAG \
$FREEZE_FLAG"

                            multi_z_script=$( cat <<EOF
#!/bin/bash
#SBATCH --job-name=multi_z_${bt_flag}_${l_flag}_${f_flag}
#SBATCH --time=$JOB_TIME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mem=${N_GB}GB
#SBATCH --output=$OUT_DIR/multi_z_${bt_flag}_${l_flag}_${f_flag}_%A_%a.out
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --mail-user=jed.homer@gmail.com
#SBATCH --array=$JOB_ARRAY_STR
#SBATCH --dependency=afterok:$sbi_deps

source /project/ls-gruen/users/jed.homer/sbi_installs/sbi_venv/bin/activate
cd /project/ls-gruen/users/jed.homer/sbiaxpdf/
mkdir -p $multi_z_log_dir
echo "Running Multi-z with global_seed=$global_seed, datavector_seed=\$SLURM_ARRAY_TASK_ID, z=\$z, bulk/tails=$bt, cumulants=$order_idx_args, pretrain=$PRETRAIN_FLAG, linearised=$LINEARISED_FLAG, freeze=$FREEZE_FLAG"
$multi_z_cmd --log_dir $multi_z_log_dir
EOF
)
                            echo "Running Multi-z with global_seed=$global_seed, datavector_seed=$SLURM_ARRAY_TASK_ID, z=$z, bulk/tails=$bt, cumulants=$order_idx_args, pretrain=$PRETRAIN_FLAG, linearised=$LINEARISED_FLAG, freeze=$FREEZE_FLAG"
                            multi_z_job_id=$(echo "$multi_z_script" | sbatch | awk '{print $4}')
                            multi_z_job_ids+=("$multi_z_job_id")
                            multi_z_deps=$( IFS=":" echo "${multi_z_job_ids[*]}" )
                        done # order_idx_args
                    done # scales
                done # bulk/tails
            done # pretrain
        done # linearised
    done # freeze
done # global_seed

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Figure One
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Run figure one after all multi_z samplings are done
figure_one_ids=()
for global_seed in $(seq 0 $N_SEEDS_GLOBAL); do
    for FREEZE_FLAG in "--freeze-parameters" "--no-freeze-parameters"; do
        for LINEARISED_FLAG in "--linearised" "--no-linearised"; do
            for PRETRAIN_FLAG in "--pre-train" "--no-pre-train"; do
                for scale_args in "${scales_sets[@]}"; do
                    for order_idx_args in "${order_idxs[@]}"; do

                        # Skip pre-train experiments
                        if [[ "$PRETRAIN_FLAG" == "--pre-train" ]]; then
                            continue
                        fi

                        # Skip non-linearised if not requested
                        if [[ "$RUN_NONLINEAR" == false && "$LINEARISED_FLAG" == "--no-linearised" ]]; then
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
python figure_one.py \
--seed $global_seed \
--seed_datavector \$SLURM_ARRAY_TASK_ID \
--n_datavectors $N_DATAVECTORS \
--n_linear_sims $N_LINEAR_SIMS \
--compression linear \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--order_idx \"$order_idx_args\" \
--scales \"$scale_args\" \
$USE_PLANCK_FLAG \
$FREEZE_FLAG"

                        # Need to run this after both bulk and tails multi_z samplings
                        figure_job=$( cat <<EOF
#!/bin/bash
#SBATCH --job-name=fig_one_${l_flag}_${f_flag}
#SBATCH --time=$JOB_TIME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mem=${N_GB}GB
#SBATCH --output=$OUT_DIR/fig_one_${l_flag}_${f_flag}_%A_%a.out
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --mail-user=jed.homer@gmail.com
#SBATCH --array=$JOB_ARRAY_STR
#SBATCH --dependency=afterok:$multi_z_deps

source /project/ls-gruen/users/jed.homer/sbi_installs/sbi_venv/bin/activate
cd /project/ls-gruen/users/jed.homer/sbiaxpdf/
mkdir -p $figure_one_log_dir
echo "Running figure one with global_seed=$global_seed, datavector_seed=\$SLURM_ARRAY_TASK_ID, cumulants=$order_idx_args, pretrain=$PRETRAIN_FLAG, linearised=$LINEARISED_FLAG, freeze=$FREEZE_FLAG"
$figure_cmd --log_dir $figure_one_log_dir
EOF
)
                        echo "Running figure one with global_seed=$global_seed, datavector_seed=$SLURM_ARRAY_TASK_ID, cumulants=$order_idx_args, pretrain=$PRETRAIN_FLAG, linearised=$LINEARISED_FLAG, freeze=$FREEZE_FLAG"
                        figure_job_id=$(echo "$figure_job" | sbatch | awk '{print $4}')
                        figure_one_ids+=("$figure_job_id")
                        figure_one_deps=$( IFS=":" echo "${figure_one_ids[*]}" )
                    done # order_idx
                done # scales
            done # pre-train
        done # linearised
    done # freeze
done # global seed

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Figure Two
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Run figure two after all figure one runs are done
for FREEZE_FLAG in "--freeze-parameters" "--no-freeze-parameters"; do
    for LINEARISED_FLAG in "--linearised" "--no-linearised"; do
        for PRETRAIN_FLAG in "--pre-train" "--no-pre-train"; do
            for scale_args in "${scales_sets[@]}"; do
                for order_idx_args in "${order_idxs[@]}"; do

                    # Skip pre-train experiments
                    if [[ "$PRETRAIN_FLAG" == "--pre-train" ]]; then
                        continue
                    fi

                    # Skip non-linearised if not requested
                    if [[ "$RUN_NONLINEAR" == false && "$LINEARISED_FLAG" == "--no-linearised" ]]; then
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
                    if [ "$LINEARISED_FLAG" == "--linearised" ]; then
                        # Label runs
                        l_flag="l"
                    else
                        l_flag="nl"
                    fi

                    # Tag for job name
                    if [ "$FREEZE_FLAG" == "--freeze-parameters" ]; then
                        # Label runs
                        f_flag="f"
                    else
                        f_flag="nf"
                    fi

                    order_idx_str=$(echo "$order_idx_args" | tr -d ' ')
                    scale_str=$(echo "$scale_args" | tr -d ' ')

                    echo ">>Running figure two with cumulants=$order_idx_args, pretrain=$PRETRAIN_FLAG, linearised=$LINEARISED_FLAG, freeze=$FREEZE_FLAG"

                    figure_two_log_dir=$(get_log_dir \
                        $BASE_LOG_DIR \
                        "figure_two" \
                        $order_idx_str \
                        $scale_str
                    )

                    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=fig_two_${l_flag}_${f_flag}
#SBATCH --time=$JOB_TIME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mem=${N_GB}GB
#SBATCH --output=$OUT_DIR/fig_two_${l_flag}_${f_flag}_%A.out
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --mail-user=jed.homer@gmail.com
#SBATCH --dependency=afterok:$figure_one_deps

source /project/ls-gruen/users/jed.homer/sbi_installs/sbi_venv/bin/activate
cd /project/ls-gruen/users/jed.homer/sbiaxpdf/
mkdir -p $figure_two_log_dir
python figure_two.py \
--n_datavectors $N_DATAVECTORS \
--n_linear_sims $N_LINEAR_SIMS \
--compression linear \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--order_idx \"$order_idx_args\" \
--scales \"$scale_args\" \
$USE_PLANCK_FLAG \
$FREEZE_FLAG \
--log_dir $figure_two_log_dir
EOF
                done # order_idx
            done # scale
        done # pre-train
    done # linearised
done # freeze