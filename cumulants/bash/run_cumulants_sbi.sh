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

FORCE_NOISELESS_DATAVECTOR=false
USE_QUIJOTE_TAILS=false # Use tails datavectors measured, not calculated, from Quijote
FIDUCIAL_REDUCE=true # Reduce cumulants with fiducial variances
USE_SCALERS=true # Not implemented yet

DEFAULT_RESOLUTION=1024

# Running a test single run or not
if [[ "$SINGLE_RUN" == "true" ]]; then
    echo "SINGLE RUN."
    N_SEEDS=1
    START_SEED=0
    N_SEEDS_GLOBAL=1    # Number of repeated trainings for SBI 
    END_SEED=$(( $START_SEED + $N_SEEDS - 1 ))
    N_PARALLEL=2

    RUN_FROZEN=false
else
    echo "MULTIPLE SEEDS RUN."
    N_SEEDS=40
    START_SEED=0
    N_SEEDS_GLOBAL=10   # Number of repeated trainings for SBI
    END_SEED=$(( $START_SEED + $N_SEEDS - 1 ))
    N_PARALLEL=100
fi

N_GB=8
N_CPU=8
JOB_TIME="02:00:00"
MAIL_TYPE="begin,end,fail"
JOB_ARRAY_STR="$START_SEED-$END_SEED%$N_PARALLEL"

N_DATAVECTORS=1       # Number of independent datavectors to sample posteriors with    
N_LINEAR_SIMS=2000      # Number of linear simulations to use for training / pre-training

order_idxs=(
    # "0"
    "0 1 2"
)

scales_sets=(
    # "15.0 20.0 25.0 30.0 35.0"
    "5.0 10.0 15.0 20.0 25.0 30.0 35.0"
)

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

# Run cumulants-from-quijote if using high resolution
data_deps=()

if [[ "$USE_QUIJOTE_TAILS" == "true" ]]; then
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

    echo \"Running cumulants data script\"

    python get_cumulants_data.py
    "

    echo "Running cumulants data scripts"

    data_job_id=$(echo "$cumulants_cmd" | sbatch | awk '{print $4}')
    data_deps+=("$data_job_id")
else
    echo "Not running cumulants data scripts"
fi

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

echo \"Running PDFs data script\"

python get_pdf_data.py
"

echo "Running PDF data scripts"

data_job_id=$(echo "$pdfs_cmd" | sbatch | awk '{print $4}')
data_deps+=("$data_job_id")
data_dep_string=$(IFS=':'; echo "${data_deps[*]}")

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
python cumulants_sbi.py \
--seed $global_seed \
--compression linear \
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
#SBATCH --output=$OUT_DIR/${bt_flag}/${l_flag}/${f_flag}/z${z}/sbi_fixed_%j.out
#SBATCH --error=$OUT_DIR/${bt_flag}/${l_flag}/${f_flag}/z${z}/sbi_fixed_%j.err
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
                    figure_one_ids=()
                    
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
python cumulants_multi_z.py \
--seed $global_seed \
--seed_datavector \$SLURM_ARRAY_TASK_ID \
--n_datavectors $N_DATAVECTORS \
--compression linear \
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
#SBATCH --output=$OUT_DIR/${global_seed}/${bt_flag}/${l_flag}/${f_flag}/m_z_%a_%j.out
#SBATCH --error=$OUT_DIR/${global_seed}/${bt_flag}/${l_flag}/${f_flag}/m_z_%a_%j.err
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
python figure_one.py \
--seed $global_seed \
--seed_datavector \$SLURM_ARRAY_TASK_ID \
--n_datavectors $N_DATAVECTORS \
--n_linear_sims $N_LINEAR_SIMS \
--compression linear \
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
#SBATCH --job-name=figure_one
#SBATCH --output=$OUT_DIR/${global_seed}/${l_flag}/${f_flag}/figure_one_%a_%j.out
#SBATCH --error=$OUT_DIR/${global_seed}/${l_flag}/${f_flag}/figure_one_%a_%j.err
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
    if [ "$LINEARISED_FLAG" == "--linearised" ]; then # Label runs
        l_flag="l"
    else
        l_flag="nl"
    fi

    # Tag for job name
    if [ "$FREEZE_FLAG" == "--freeze-parameters" ]; then # Label runs
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

    sbatch <<END
#!/bin/bash
#SBATCH --job-name=figure_two
#SBATCH --output=$OUT_DIR/${l_flag}/${f_flag}/figure_two_%j.out
#SBATCH --error=$OUT_DIR/${l_flag}/${f_flag}/figure_two_%j.err
#SBATCH --partition=cluster
#SBATCH --time=$JOB_TIME
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --dependency=afterok:$data_dep_string:$sbi_deps:$multi_z_deps:$figure_one_deps

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

echo "Running final figure two script"

python figure_two.py \
--n_datavectors $N_DATAVECTORS \
--compression linear \
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