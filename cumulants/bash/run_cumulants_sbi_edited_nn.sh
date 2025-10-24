#!/bin/bash

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Run `N_SEEDS_GLOBAL` repeated cumulants-SBI experiments for different redshifts and scales
# where `N_SEEDS` independent datavectors are drawn to sample posteriors with 
# Adds DRY_RUN support and ensures Figure 1 waits for BOTH bulk & tails multi_z for a seed.
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

# ------------- DRY RUN SUPPORT -------------
# Set DRY_RUN=true to avoid submitting jobs; fake job IDs will be generated and printed.
: "${DRY_RUN:=true}"
_FAKE_JID=700000
_next_fake_jid() { _FAKE_JID=$((_FAKE_JID+1)); echo "${_FAKE_JID}"; }
submit_block_and_get_id() {
    # Usage: submit_block_and_get_id "$block_string"
    local block="$1"
    if [[ "$DRY_RUN" == "true" ]]; then
        local jid="$(_next_fake_jid)"
        echo "DRY-RUN: would submit job -> assigning fake id ${jid}"
        # mimic sbatch output: "Submitted batch job <id>"
        echo "Submitted batch job ${jid}"
    else
        echo "$block" | sbatch
    fi
}
submit_here_doc_and_get_id() {
    # Usage:
    # jid=$(submit_here_doc_and_get_id "$(cat <<'END'\n... \nEND\n)")
    # This function simply proxies to submit_block_and_get_id for symmetry.
    submit_block_and_get_id "$1"
}
# ------------- END DRY RUN SUPPORT -------------

# Command line args
RESULTS_DIR="results/${1:-"results/"}"
SINGLE_RUN="${2:-false}" # Run all experiments once for a single figure one
ONLY_RUN_FIGURES="${3:-false}" # Only run figure_one.py jobs

RUN_LINEARISED=true
RUN_NONLINEAR=true

DEFAULT_NDE_TYPE="CNF"
DEFAULT_N_NDES=2

NON_GAUSSIAN_TEST=false
FORCE_RECOMPUTE_DATASET=false

COMPRESSION="nn"
N_ENSEMBLE_NETS=4
NN_TYPE="NN"

USE_PRECISION_NN=false
COVARIANCE_NN=true
DATA_PROCESS_TYPE_NN="dp" #"d"
NN_CLIP_NORM=true # Global clipping of weights

USE_SOBOL=true
FORCE_NOISELESS_DATAVECTOR=false
FIDUCIAL_REDUCE=true # Reduce cumulants with fiducial variances
DEFAULT_RESOLUTION=1024
RECALCULATE_DATASETS=false
DATASETS_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/quijote_data/datasets/"
DATASET_TEST="False" # Never use this test dataset flag here

N_DATAVECTORS=1 # Number of independent datavectors to sample posteriors with (with N_SEEDS different posteriors)

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
    N_SEEDS_GLOBAL=1 # 9 # Number of repeated trainings for SBI
    END_SEED=$(( $START_SEED + $N_SEEDS - 1 ))
    N_PARALLEL=100
fi

# Reusable env-exports block (no mkdir here)
export LOG_LEVEL="$LOG_LEVEL"
export RESULTS_DIR="$RESULTS_DIR"

export DEFAULT_NDE_TYPE="$DEFAULT_NDE_TYPE"
export DEFAULT_N_NDES="$DEFAULT_N_NDES"

export N_DATAVECTOR_SEEDS=$N_SEEDS 
export N_REPEATED_SBI_SEEDS=$N_SEEDS_GLOBAL 
export FORCE_NOISELESS_DATAVECTOR="$FORCE_NOISELESS_DATAVECTOR"
export FIDUCIAL_REDUCE="$FIDUCIAL_REDUCE"
export DEFAULT_RESOLUTION="$DEFAULT_RESOLUTION"
export NON_GAUSSIAN_TEST="$NON_GAUSSIAN_TEST"
export USE_SOBOL="$USE_SOBOL"
export FORCE_RECOMPUTE_DATASET="$FORCE_RECOMPUTE_DATASET"

export DATA_PROCESS_TYPE_NN="$DATA_PROCESS_TYPE_NN"
export USE_PRECISION_NN="$USE_PRECISION_NN"
export COVARIANCE_NN="$COVARIANCE_NN"
export N_ENSEMBLE_NETS="$N_ENSEMBLE_NETS"
export NN_TYPE="$NN_TYPE"
export NN_CLIP_NORM="$NN_CLIP_NORM"

LOG_LEVEL="DEBUG"
USE_TQDM="--use-tqdm"

# JAX cache stuff
JAX_CACHE_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/jax_cache"
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_DIR"
export JAX_LOG_COMPILES=1

rm -rf "${JAX_CACHE_DIR:?}/"*
echo "Emptied JAX cache dir: $JAX_CACHE_DIR"

# SLURM job parameters
N_GB=12
N_CPU=8
JOB_TIME="24:00:00"
MAIL_TYPE="begin,end,fail"
JOB_ARRAY_STR="$START_SEED-$END_SEED%$N_PARALLEL"

# Datavector setup
order_idxs=(
    # "0"
    "0 1 2"
)

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

#############################################################
#############################################################
############# GET PDF AND/OR CUMULANTS DATA #################
#############################################################
#############################################################

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

mkdir -p \"$cumulants_data_log_dir\"

export LOG_DIR=\"${cumulants_data_log_dir}/\"

echo \"Running cumulants data script\"

uv run python get_cumulants_data.py
"

    echo "Submitting cumulants data scripts"
    data_job_id=$(submit_block_and_get_id "$cumulants_cmd" | awk '{print $4}')
    data_deps+=("$data_job_id")
else
    echo "Not running cumulants data scripts"
fi

#############################################################
#############################################################
############## RUN NON-SOBOL DATA JOBS ######################
#############################################################
#############################################################

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
#SBATCH --export=ALL

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/data/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

mkdir -p \"$pdfs_data_log_dir\"

export LOG_DIR=\"${pdfs_data_log_dir}/\"

echo \"Running PDFs data script\"

uv run python get_pdf_data.py
"

    echo "Submitting PDF data scripts"
    data_job_id=$(submit_block_and_get_id "$pdfs_cmd" | awk '{print $4}')
    data_deps+=("$data_job_id")
fi

# If not running data jobs, quickly submit an empty job
# to keep the data_dep_string non-empty
if [ ${#data_deps[@]} -eq 0 ]; then
    dummy_job_block=$'#!/bin/bash\n#SBATCH --job-name=dummy\n#SBATCH --time=00:01:00\n#SBATCH --mem=1M\ntrue\n'
    dummy_job_id=$(submit_block_and_get_id "$dummy_job_block" | awk '{print $4}')
    data_deps=("$dummy_job_id")
fi
# Always build data_dep_string from the array
data_dep_string=$(IFS=':'; echo "${data_deps[*]}")


#############################################################
#############################################################
############## RUN SBI AND MULTI-Z SAMPLING #################
#############################################################
#############################################################


# Track all Figure 1 job IDs globally so Figure 2 can depend on them if desired
FIG1_JOB_IDS=()

# Run SBI + multi_z + figure one with correct dependencies
for global_seed in $(seq 0 $N_SEEDS_GLOBAL); do
    for LINEARISED_FLAG in "--linearised" "--no-linearised"; do
        for PRETRAIN_FLAG in "--pre-train" "--no-pre-train"; do

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

            # Labels for job names
            if [ "$LINEARISED_FLAG" == "--linearised" ]; then l_flag="l"; else l_flag="nl"; fi

            for scale_args in "${scales_sets[@]}"; do
            for order_idx_args in "${order_idxs[@]}"; do

                order_idx_str=$(echo "$order_idx_args" | tr -d ' ')
                scale_str=$(echo "$scale_args" | tr -d ' ')
                redshifts_str="${all_redshifts[*]}"
                
                # We'll collect multi_z job ids across BOTH bulk and tails to enforce the Figure 1 dependency.
                multi_z_job_ids=()
                # Also track all SBI jobs across bulk/tails to optionally require for Fig1.
                sbi_job_ids_all=()

                for bt in "bulk" "tails"; do

                    # Tag for job name
                    if [ "$bt" == "bulk" ]; then bt_flag="b"; else bt_flag="t"; fi


                    #############################################################
                    #################### RUN NN (pre-SBI) ######################
                    #############################################################

                    # Submit NN per redshift for this bt, then make SBI depend on it
                    sbi_job_ids=()
                    for z in "${all_redshifts[@]}"; do
                        echo ">>Running NN with z=$z, bulk/tails=$bt, cumulants=$order_idx_args, pretrain=$PRETRAIN_FLAG, linearised=$LINEARISED_FLAG"

                        nn_log_dir=$(get_log_dir \
                            $BASE_LOG_DIR \
                            "nn" \
                            $z \
                            $bt \
                            $order_idx_str \
                            $scale_str \
                            $LINEARISED_FLAG \
                            $PRETRAIN_FLAG \
                            $global_seed
                        )

                        nn_cmd_inner="uv run python cumulants_nn.py \
--seed $global_seed \
--compression $COMPRESSION \
--n_datavectors $N_DATAVECTORS \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--n_linear_sims $N_LINEAR_SIMS \
--order_idx $order_idx_args \
--scales $scale_args \
--redshift $z \
--no-use-tqdm \
--bulk_or_tails $bt"

                        nn_job_script=$(
                            cat <<END
#!/bin/bash
#SBATCH --job-name=nn_${global_seed}_${bt_flag}_${l_flag}_z${z}
#SBATCH --output=$OUT_DIR/nn/${bt_flag}/${l_flag}/z${z}/nn_${global_seed}.out
#SBATCH --error=$OUT_DIR/nn/${bt_flag}/${l_flag}/z${z}/nn_${global_seed}.err
#SBATCH --partition=cluster
#SBATCH --time=$JOB_TIME
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --dependency=afterok:$data_dep_string
#SBATCH --export=ALL

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

mkdir -p "$nn_log_dir"

export LOG_DIR="$nn_log_dir"

echo "Running NN script with seed $global_seed and redshift $z"
$nn_cmd_inner
END
                        )

                        # Submit NN and capture its job id
                        nn_job_id=$(submit_block_and_get_id "$nn_job_script" | awk '{print $4}')

                        #############################################################
                        ######################### RUN SBI ###########################
                        #############################################################

                        echo ">>Running SBI with z=$z, bulk/tails=$bt, cumulants=$order_idx_args, pretrain=$PRETRAIN_FLAG, linearised=$LINEARISED_FLAG"

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

                        sbi_cmd_inner="uv run python cumulants_sbi.py \
--seed $global_seed \
--compression $COMPRESSION \
--n_datavectors $N_DATAVECTORS \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--n_linear_sims $N_LINEAR_SIMS \
--order_idx $order_idx_args \
--scales $scale_args \
--redshift $z \
--no-use-tqdm \
--bulk_or_tails $bt"

                        sbi_job_script=$(
                            cat <<END
#!/bin/bash
#SBATCH --job-name=sbi_${global_seed}_${bt_flag}_${l_flag}_z${z}
#SBATCH --output=$OUT_DIR/sbi/${bt_flag}/${l_flag}/z${z}/sbi_${global_seed}.out
#SBATCH --error=$OUT_DIR/sbi/${bt_flag}/${l_flag}/z${z}/sbi_${global_seed}.err
#SBATCH --partition=cluster
#SBATCH --time=$JOB_TIME
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --dependency=afterok:$data_dep_string:$nn_job_id
#SBATCH --export=ALL

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

mkdir -p "$sbi_log_dir"

export LOG_DIR="$sbi_log_dir"

echo "Running sbi script with seed $global_seed and redshift $z"
$sbi_cmd_inner
END
                        )

                        sbi_job_id=$(submit_block_and_get_id "$sbi_job_script" | awk '{print $4}')
                        sbi_job_ids+=("$sbi_job_id")
                        sbi_job_ids_all+=("$sbi_job_id")
                    done # z

                    # SBI deps for this bt across all redshifts
                    sbi_deps=$(IFS=":"; echo "${sbi_job_ids[*]}")


                    #############################################################
                    ####################### RUN MULTI-Z #########################
                    #############################################################


                    # multi-z per bt depends on *all* its SBI redshift jobs
                    if [[ "$ONLY_RUN_FIGURES" == false ]]; then

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
                        multi_z_cmd_inner="uv run python cumulants_multi_z.py \
--seed $global_seed \
--seed_datavector \$SLURM_ARRAY_TASK_ID \
--n_datavectors $N_DATAVECTORS \
--n_linear_sims $N_LINEAR_SIMS \
--compression $COMPRESSION \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--bulk_or_tails $bt \
--order_idx $order_idx_args \
--scales $scale_args \
--redshifts $redshifts_str"

                        multi_z_script=$(
                            cat <<END
#!/bin/bash
#SBATCH --job-name=m_z_${global_seed}_${bt_flag}_${l_flag}
#SBATCH --output=$OUT_DIR/multi_z/${global_seed}/${bt_flag}/${l_flag}/m_z_${global_seed}_%a_%j.out
#SBATCH --error=$OUT_DIR/multi_z/${global_seed}/${bt_flag}/${l_flag}/m_z_${global_seed}_%a_%j.err
#SBATCH --array=$JOB_ARRAY_STR
#SBATCH --partition=cluster
#SBATCH --time=$JOB_TIME
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --dependency=afterok:$data_dep_string:$sbi_deps
#SBATCH --export=ALL

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

mkdir -p "${multi_z_log_dir}/${SLURM_ARRAY_TASK_ID}"

export LOG_DIR="${multi_z_log_dir}/${SLURM_ARRAY_TASK_ID}"

echo "Running final multi-z script"
$multi_z_cmd_inner
END
                        )

                        echo ">>Submitting Multi-z for $bt with global_seed=$global_seed"
                        multi_z_job_id=$(submit_block_and_get_id "$multi_z_script" | awk '{print $4}')
                        multi_z_job_ids+=("$multi_z_job_id")
                    fi # ONLY_RUN_FIGURES

                done # bt in bulk tails

                # Build Figure 1 dependency that requires BOTH bulk & tails multi-z to finish,
                # and (optionally) all SBI jobs as well (redundant, but safe).
                multi_z_deps=$(IFS=":"; echo "${multi_z_job_ids[*]}")
                sbi_deps_all=$(IFS=":"; echo "${sbi_job_ids_all[*]}")

                figure_one_log_dir=$(get_log_dir \
                    $BASE_LOG_DIR \
                    "figure_one" \
                    both_bt \
                    $order_idx_str \
                    $scale_str \
                    $LINEARISED_FLAG \
                    $PRETRAIN_FLAG \
                    $global_seed
                )

                figure_cmd="uv run python figure_one.py \
--seed $global_seed \
--seed_datavector \$SLURM_ARRAY_TASK_ID \
--n_datavectors $N_DATAVECTORS \
--n_linear_sims $N_LINEAR_SIMS \
--compression $COMPRESSION \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--order_idx $order_idx_args \
--scales $scale_args \
--redshifts $redshifts_str"

                figure_job=$(
                    cat <<END
#!/bin/bash
#SBATCH --job-name=figure_1_${global_seed}_${l_flag}
#SBATCH --output=$OUT_DIR/figure_1/${global_seed}/${l_flag}/figure_one_${global_seed}_%a_%j.out
#SBATCH --error=$OUT_DIR/figure_1/${global_seed}/${l_flag}/figure_one_${global_seed}_%a_%j.err
#SBATCH --array=$JOB_ARRAY_STR
#SBATCH --partition=cluster
#SBATCH --time=$JOB_TIME
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --dependency=afterok:$data_dep_string:$sbi_deps_all:$multi_z_deps
#SBATCH --export=ALL

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

mkdir -p "${figure_one_log_dir}/${SLURM_ARRAY_TASK_ID}"

export LOG_DIR="${figure_one_log_dir}/${SLURM_ARRAY_TASK_ID}"

echo "Running final figure one script"
$figure_cmd
END
                )

                echo ">>Submitting Figure One for global_seed=$global_seed (depends on BOTH bulk & tails multi_z)"
                figure_job_id=$(submit_block_and_get_id "$figure_job" | awk '{print $4}')
                FIG1_JOB_IDS+=("$figure_job_id")

            done
            done # scales
        done
    done
done

# Build colon-separated deps for all Figure 1 jobs (used by Figure 2 if desired)
if [[ ${#FIG1_JOB_IDS[@]} -gt 0 ]]; then
    figure_one_deps_all="$(IFS=:; echo "${FIG1_JOB_IDS[*]}")"
else
    figure_one_deps_all=""
fi
export figure_one_deps_all
echo "figure_one_deps_all=$figure_one_deps_all"


#############################################################
#############################################################
###################### RUN FIGURE TWO #######################
#############################################################
#############################################################


# We keep your original Figure 2 submission loop, but we can (optionally) make it wait
# for all Figure 1 jobs by adding figure_one_deps_all to --dependency.
# If you prefer the old behavior, set REQUIRE_FIG1_FOR_FIG2=false.
: "${REQUIRE_FIG1_FOR_FIG2:=true}"
FIG2_JOB_IDS=()

redshifts_str="${all_redshifts[*]}"

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

    # Labels for job name
    if [[ "$LINEARISED_FLAG" == "--linearised" ]]; then
        l_flag="l"
    else
        l_flag="nl"
    fi

    order_idx_str=$(echo "$order_idx_args" | tr -d ' ')
    scale_str=$(echo "$scale_args" | tr -d ' ')

    echo ">>Submitting Figure 2 with cumulants=$order_idx_args, linearised=$LINEARISED_FLAG"

    figure_two_log_dir=$(get_log_dir \
        "$BASE_LOG_DIR" \
        "figure_two" \
        "$order_idx_str" \
        "$scale_str" \
    )

    dep_line="$data_dep_string"
    if [[ "$REQUIRE_FIG1_FOR_FIG2" == "true" && -n "$figure_one_deps_all" ]]; then
        dep_line="${dep_line}:$figure_one_deps_all"
    fi

    fig2_block=$(
        cat <<END
#!/bin/bash
#SBATCH --job-name=fig2_${l_flag}_${order_idx_str}_${scale_str}
#SBATCH --output=$OUT_DIR/figure_two/fig2_%j.out
#SBATCH --error=$OUT_DIR/figure_two/fig2_%j.err
#SBATCH --partition=cluster
#SBATCH --time=$JOB_TIME
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --dependency=afterok:$dep_line
#SBATCH --export=ALL

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

export LOG_DIR="${figure_two_log_dir}/"

echo "Running Figure 2 script"

uv run python figure_two.py \
--n_datavectors $N_DATAVECTORS \
--compression $COMPRESSION \
--n_linear_sims $N_LINEAR_SIMS \
--order_idx $order_idx_args \
--scales $scale_args \
--redshifts $redshifts_str \
$LINEARISED_FLAG \
$PRETRAIN_FLAG

END
    )

    sbatch_out=$(submit_block_and_get_id "$fig2_block")
    jid=$(echo "$sbatch_out" | awk '{print $4}')
    if [[ -n "$jid" ]]; then
        FIG2_JOB_IDS+=("$jid")
        echo "   -> Figure 2 job submitted: $jid"
    else
        echo "   !! Failed to capture job ID for Figure 2 submission"
    fi

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


#############################################################
#############################################################
####################### RUN COVERAGES #######################
#############################################################
#############################################################


for PRETRAIN_FLAG in "--pre-train" "--no-pre-train"; do
for scale_args in "${scales_sets[@]}"; do
for order_idx_args in "${order_idxs[@]}"; do

    # Skip pre-train experiments
    if [[ "$PRETRAIN_FLAG" == "--pre-train" ]]; then
        continue
    fi

    order_idx_str=$(echo "$order_idx_args" | tr -d ' ')
    scale_str=$(echo "$scale_args" | tr -d ' ')
 
    echo ">>Submitting COVERAGES with cumulants=$order_idx_args, pretrain=$PRETRAIN_FLAG"

    figure_two_log_dir=$(get_log_dir \
        $BASE_LOG_DIR \
        "figure_two" \
        $order_idx_str \
        $scale_str
    )

    cov_block=$(
        cat <<END
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
#SBATCH --export=ALL

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

export LOG_DIR="${figure_two_log_dir}/"

echo "Running final coverages script"

uv run python figure_coverages.py 

END
    )
    submit_block_and_get_id "$cov_block" >/dev/null

done
done
done