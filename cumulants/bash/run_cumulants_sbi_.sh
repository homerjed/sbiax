#!/bin/bash
# run_cumulants_sbi_.sh
set -euo pipefail

# ============================ helpers ============================
get_log_dir() {
  local results_dir="$1"; shift
  local joined=""
  for part in "$@"; do joined="${joined}/${part}"; done
  echo "${results_dir%/}$joined"
}

join_by_colon() { local IFS=":"; echo "$*"; }

dep_line_or_empty() {
  local deps="$1"
  if [[ -n "${deps// /}" ]]; then
    echo "#SBATCH --dependency=afterok:$deps"
  fi
}

# -------- Plan recording (for dry-run & real) --------
PLAN_LABELS=()
PLAN_IDS=()
PLAN_DEPS=()
plan_record() { PLAN_LABELS+=("$1"); PLAN_IDS+=("$2"); PLAN_DEPS+=("$3"); }
plan_dump() {
  local tsv="$OUT_DIR/plan.tsv"
  local dot="$OUT_DIR/plan.dot"
  mkdir -p "$OUT_DIR"

  echo -e "JobLabel\tJobID\tDependencies" > "$tsv"
  for ((i=0; i<${#PLAN_LABELS[@]}; i++)); do
    echo -e "${PLAN_LABELS[$i]}\t${PLAN_IDS[$i]}\t${PLAN_DEPS[$i]}" >> "$tsv"
  done

  {
    echo "digraph DAG {"
    echo "  rankdir=LR;"
    echo "  node [shape=box,fontsize=10];"
    for ((i=0; i<${#PLAN_LABELS[@]}; i++)); do
      lbl="${PLAN_LABELS[$i]}"; id="${PLAN_IDS[$i]}"; deps="${PLAN_DEPS[$i]}"
      echo "  \"${id}\\n${lbl}\";"
      if [[ -n "$deps" ]]; then
        IFS=':' read -r -a dlist <<< "$deps"
        for d in "${dlist[@]}"; do
          [[ -z "$d" ]] && continue
          echo "  \"${d}\" -> \"${id}\\n${lbl}\";"
        done
      fi
    done
    echo "}"
  } > "$dot"

  echo "Wrote plan to:"
  echo "  $tsv"
  echo "  $dot  (render with: dot -Tpng \"$dot\" -o plan.png)"
}

# -------- job submission wrapper (file-based; dry-run aware) --------
JOB_COUNTER=50000
submit_job() {
  local job_script="$1"
  local job_label="$2"
  local dry_run="$3"
  local deps_line="${4:-}"  # colon-joined deps used by this job

  local scripts_dir="$OUT_DIR/scripts"
  mkdir -p "$scripts_dir"

  local ts; ts=$(date +%s)
  local script_path="${scripts_dir}/${job_label//[^A-Za-z0-9_.-]/_}_${ts}_$$.sh"

  umask 077
  printf "%s\n" "$job_script" > "$script_path"
  chmod +x "$script_path"

  if [[ "$dry_run" == "true" ]]; then
    JOB_COUNTER=$((JOB_COUNTER + 1))
    local fake_id="$JOB_COUNTER"
    echo "DRY-RUN: $job_label  id=${fake_id}  deps=${deps_line:-<none>}  file=$script_path"
    awk '/^#SBATCH/ || /^echo "Running/ || /^python /' "$script_path" || true
    plan_record "$job_label" "$fake_id" "${deps_line}"
    echo "$fake_id"
  else
    local real_id
    real_id=$(sbatch --parsable "$script_path")
    echo "SUBMITTED: $job_label  id=${real_id}  deps=${deps_line:-<none>}  file=$script_path"
    plan_record "$job_label" "$real_id" "${deps_line}"
    echo "$real_id"
  fi
}

# ============================ args & flags ============================
RESULTS_DIR="results/${1:-"results/"}"
SINGLE_RUN="${2:-false}"
ONLY_RUN_FIGURES="${3:-false}"
DRY_RUN="${4:-false}"  # NEW

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
N_ENSEMBLE_NETS=20
USE_PRECISION_NN=false
DATA_PROCESS_TYPE_NN="d"

USE_SOBOL=true
FORCE_NOISELESS_DATAVECTOR=false
USE_QUIJOTE_TAILS=false
FIDUCIAL_REDUCE=true
DEFAULT_RESOLUTION=1024

RECALCULATE_DATASETS=false
DATASETS_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/quijote_data/datasets/"

N_DATAVECTORS=1
AFFINE_SAMPLE=false
BLACKJAX_SAMPLE=true

if [[ "$SINGLE_RUN" == "true" ]]; then
  echo "SINGLE RUN."
  N_SEEDS=1
  START_SEED=0
  N_SEEDS_GLOBAL=1
  END_SEED=$(( START_SEED + N_SEEDS - 1 ))
  N_PARALLEL=2
else
  echo "MULTIPLE SEEDS RUN."
  N_SEEDS=50
  START_SEED=0
  N_SEEDS_GLOBAL=1
  END_SEED=$(( START_SEED + N_SEEDS - 1 ))
  N_PARALLEL=100
fi

N_GB=12
N_CPU=8
JOB_TIME="16:00:00"
MAIL_TYPE="begin,end,fail"
JOB_ARRAY_STR="$START_SEED-$END_SEED%$N_PARALLEL"

order_idxs=("0 1 2")

if [[ "$USE_SOBOL" == "true" ]]; then
  scales_sets=("5.9 9.8 13.7 17.6 21.5 25.4 29.3 33.2")
  N_LINEAR_SIMS=32768
else
  scales_sets=("5.0 10.0 15.0 20.0 25.0 30.0 35.0")
  N_LINEAR_SIMS=2000
fi

all_redshifts=(0.0 0.5 1.0)
redshifts_str="${all_redshifts[*]}"

TIMESTAMP=$(date +'%m%d_%H%M')
OUT_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/cumulants_sbi/$TIMESTAMP"
mkdir -p "$OUT_DIR"

if [[ "$RECALCULATE_DATASETS" == "true" ]]; then
  rm -rf "${DATASETS_DIR:?}/"*
  echo "Emptied datasets directory: $DATASETS_DIR"
else
  echo "Didn't empty datasets directory: $DATASETS_DIR"
fi

BASE_LOG_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/$RESULTS_DIR/logs/"
mkdir -p "$BASE_LOG_DIR"

LOG_LEVEL="DEBUG"
USE_TQDM="--use-tqdm"
USE_PLANCK_FLAG=$([[ "$USE_PLANCK" == "true" ]] && echo "--use-planck" || echo "--no-use-planck")

# ============================ data jobs ============================
data_deps=()

if [[ "$USE_QUIJOTE_TAILS" == "true" && "$USE_SOBOL" == "false" ]]; then
  cumulants_data_log_dir=$(get_log_dir "$BASE_LOG_DIR" "cumulants_data"); mkdir -p "$cumulants_data_log_dir"
  cumulants_cmd=$(cat <<EOF
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

echo "Running cumulants data script"
python get_cumulants_data.py
EOF
)
  data_job_id=$(submit_job "$cumulants_cmd" "cumulants_data" "$DRY_RUN" "")
  data_deps+=("$data_job_id")
else
  echo "Not running cumulants data scripts"
fi

if [[ "$USE_SOBOL" == "false" ]]; then
  pdfs_data_log_dir=$(get_log_dir "$BASE_LOG_DIR" "pdfs_data"); mkdir -p "$pdfs_data_log_dir"
  pdfs_cmd=$(cat <<EOF
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

echo "Running PDFs data script"
python get_pdf_data.py
EOF
)
  data_job_id=$(submit_job "$pdfs_cmd" "pdfs_data" "$DRY_RUN" "")
  data_deps+=("$data_job_id")
else
  echo "Not running PDF data scripts"
fi

data_dep_string=""
if ((${#data_deps[@]} > 0)); then
  data_dep_string=$(join_by_colon "${data_deps[@]}")
fi

# ============================ main jobs ============================
for (( global_seed=0; global_seed<${N_SEEDS_GLOBAL}; global_seed++ )); do
for FREEZE_FLAG in "--freeze-parameters" "--no-freeze-parameters"; do
for LINEARISED_FLAG in "--linearised" "--no-linearised"; do
for PRETRAIN_FLAG in "--pre-train" "--no-pre-train"; do

  if [[ "$PRETRAIN_FLAG" == "--pre-train" ]]; then continue; fi
  if [[ "$RUN_LINEARISED" == false && "$LINEARISED_FLAG" == "--linearised" ]]; then echo "Skip linearised"; continue; fi
  if [[ "$RUN_NONLINEAR" == false && "$LINEARISED_FLAG" == "--no-linearised" ]]; then echo "Skip non-linearised"; continue; fi
  if [[ "$RUN_FROZEN" == false && "$FREEZE_FLAG" == "--freeze-parameters" ]]; then continue; fi
  if [[ "$FREEZE_FLAG" == "--freeze-parameters" && "$USE_PLANCK" == true ]]; then continue; fi

  for scale_args in "${scales_sets[@]}"; do
  for order_idx_args in "${order_idxs[@]}"; do

    sbi_job_ids_all_bt=()
    multi_z_job_ids_all_bt=()

    for bt in "bulk" "tails"; do
      [[ "$bt" == "bulk" ]] && bt_flag="b" || bt_flag="t"
      [[ "$LINEARISED_FLAG" == "--linearised" ]] && l_flag="l" || l_flag="nl"
      [[ "$FREEZE_FLAG" == "--freeze-parameters" ]] && f_flag="f" || f_flag="nf"

      order_idx_str=$(echo "$order_idx_args" | tr -d ' ')
      scale_str=$(echo "$scale_args" | tr -d ' ')

      # ---------- SBI per z ----------
      sbi_job_ids=()
      for z in "${all_redshifts[@]}"; do
        sbi_log_dir=$(get_log_dir "$BASE_LOG_DIR" "sbi" "$z" "$bt" "$order_idx_str" "$scale_str" "$LINEARISED_FLAG" "$PRETRAIN_FLAG" "$global_seed"); mkdir -p "$sbi_log_dir"

        sbi_cmd="python cumulants_sbi.py \
--seed $global_seed \
--compression $COMPRESSION \
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

        dep_bits=()
        [[ -n "$data_dep_string" ]] && dep_bits+=("$data_dep_string")
        sbi_resolved_deps="$(join_by_colon "${dep_bits[@]}")"
        sbi_dep_line="$(dep_line_or_empty "$sbi_resolved_deps")"

        sbi_job_script=$(cat <<EOF
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
${sbi_dep_line}

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

echo "Running sbi script with seed $global_seed and redshift $z"
$sbi_cmd
EOF
)
        echo ">>Scheduling SBI z=$z bt=$bt lin=$LINEARISED_FLAG freeze=$FREEZE_FLAG seed=$global_seed"
        sbi_job_id=$(submit_job "$sbi_job_script" "sbi_${global_seed}_${bt_flag}_${l_flag}_${f_flag}_z${z}" "$DRY_RUN" "$sbi_resolved_deps")
        sbi_job_ids+=("$sbi_job_id")
      done

      sbi_deps=$(join_by_colon "${sbi_job_ids[@]}")
      sbi_job_ids_all_bt+=("${sbi_job_ids[@]}")

      # ---------- multi-z ----------
      if [[ "$ONLY_RUN_FIGURES" == "false" ]]; then
        multi_z_log_dir=$(get_log_dir "$BASE_LOG_DIR" "multi_z" "$bt" "$order_idx_str" "$scale_str" "$LINEARISED_FLAG" "$PRETRAIN_FLAG" "$global_seed")

        multi_z_cmd="python cumulants_multi_z.py \
--seed $global_seed \
--seed_datavector \$SLURM_ARRAY_TASK_ID \
--n_datavectors $N_DATAVECTORS \
--compression $COMPRESSION \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--n_linear_sims $N_LINEAR_SIMS \
--order_idx \"$order_idx_args\" \
--scales \"$scale_args\" \
--bulk_or_tails $bt \
--redshifts $redshifts_str \
$USE_PLANCK_FLAG \
$FREEZE_FLAG"

        dep_bits=()
        [[ -n "$data_dep_string" ]] && dep_bits+=("$data_dep_string")
        [[ -n "$sbi_deps" ]] && dep_bits+=("$sbi_deps")
        multi_z_resolved_deps="$(join_by_colon "${dep_bits[@]}")"
        multi_z_dep_line="$(dep_line_or_empty "$multi_z_resolved_deps")"

        multi_z_script=$(cat <<EOF
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
${multi_z_dep_line}

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

mkdir -p "${multi_z_log_dir}/\${SLURM_ARRAY_TASK_ID}"

export LOG_DIR="${multi_z_log_dir}/\${SLURM_ARRAY_TASK_ID}"
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

echo "Running final multi-z script"
$multi_z_cmd
EOF
)
        echo ">>Scheduling Multi-z bt=$bt lin=$LINEARISED_FLAG freeze=$FREEZE_FLAG seed=$global_seed"
        multi_z_job_id=$(submit_job "$multi_z_script" "m_z_${global_seed}_${bt_flag}_${l_flag}_${f_flag}" "$DRY_RUN" "$multi_z_resolved_deps")
        multi_z_job_ids_all_bt+=("$multi_z_job_id")
      fi
    done  # bt

    # ---------- figure one (depends on BOTH bulk & tails multi-z) ----------
    order_idx_str=$(echo "$order_idx_args" | tr -d ' ')
    scale_str=$(echo "$scale_args" | tr -d ' ')
    figure_one_log_dir=$(get_log_dir "$BASE_LOG_DIR" "figure_one" "both_bt" "$order_idx_str" "$scale_str" "$LINEARISED_FLAG" "$PRETRAIN_FLAG" "$global_seed")

    figure_cmd="python figure_one.py \
--seed $global_seed \
--seed_datavector \$SLURM_ARRAY_TASK_ID \
--n_datavectors $N_DATAVECTORS \
--n_linear_sims $N_LINEAR_SIMS \
--compression $COMPRESSION \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--order_idx \"$order_idx_args\" \
--scales \"$scale_args\" \
--redshifts $redshifts_str \
$USE_PLANCK_FLAG \
$FREEZE_FLAG"

    sbi_deps_all=$(join_by_colon "${sbi_job_ids_all_bt[@]}")
    multi_z_deps_all=$(join_by_colon "${multi_z_job_ids_all_bt[@]}")

    dep_bits=()
    [[ -n "$data_dep_string" ]] && dep_bits+=("$data_dep_string")
    [[ -n "$sbi_deps_all" ]] && dep_bits+=("$sbi_deps_all")
    [[ -n "$multi_z_deps_all" ]] && dep_bits+=("$multi_z_deps_all")
    figure_resolved_deps="$(join_by_colon "${dep_bits[@]}")"
    figure_dep_line="$(dep_line_or_empty "$figure_resolved_deps")"

    [[ "$LINEARISED_FLAG" == "--linearised" ]] && l_flag="l" || l_flag="nl"
    [[ "$FREEZE_FLAG" == "--freeze-parameters" ]] && f_flag="f" || f_flag="nf"

    figure_job=$(cat <<EOF
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
${figure_dep_line}

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

mkdir -p "${figure_one_log_dir}/\${SLURM_ARRAY_TASK_ID}"

export LOG_DIR="${figure_one_log_dir}/\${SLURM_ARRAY_TASK_ID}"
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

echo "Running final figure one script"
$figure_cmd
EOF
)
    echo ">>Scheduling figure one seed=$global_seed lin=$LINEARISED_FLAG freeze=$FREEZE_FLAG"
    figure_job_id=$(submit_job "$figure_job" "figure_1_${global_seed}" "$DRY_RUN" "$figure_resolved_deps")
    figure_one_deps="$figure_job_id"

    # ---------- figure two (after data + SBIs + multi-z + fig1) ----------
    figure_two_log_dir=$(get_log_dir "$BASE_LOG_DIR" "figure_two" "$order_idx_str" "$scale_str"); mkdir -p "$figure_two_log_dir"

    dep_bits=()
    [[ -n "$data_dep_string" ]] && dep_bits+=("$data_dep_string")
    [[ -n "$sbi_deps_all"   ]] && dep_bits+=("$sbi_deps_all")
    [[ -n "$multi_z_deps_all" ]] && dep_bits+=("$multi_z_deps_all")
    [[ -n "$figure_one_deps" ]] && dep_bits+=("$figure_one_deps")
    figure2_resolved_deps="$(join_by_colon "${dep_bits[@]}")"
    figure_two_dep_line="$(dep_line_or_empty "$figure2_resolved_deps")"

    figure_two_job=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=figure_2
#SBATCH --output=$OUT_DIR/figure_2/${l_flag}/${f_flag}/figure_two_%j.out
#SBATCH --error=$OUT_DIR/figure_2/${l_flag}/${f_flag}/figure_two_%j.err
#SBATCH --partition=cluster
#SBATCH --time=$JOB_TIME
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=$MAIL_TYPE
${figure_two_dep_line}

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
export DATA_PROCESS_TYPE_NN="$DATA_PROCESS_TYPE_NN"
export USE_PRECISION_NN="$USE_PRECISION_NN"
export USE_SCALERS="$USE_SCALERS"
export BLACKJAX_SAMPLE="$BLACKJAX_SAMPLE"
export AFFINE_SAMPLE="$AFFINE_SAMPLE"
export N_ENSEMBLE_NETS="$N_ENSEMBLE_NETS"

echo "Running final figure two script"
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
EOF
)
    echo ">>Scheduling figure two seed=$global_seed lin=$LINEARISED_FLAG freeze=$FREEZE_FLAG"
    _=$(submit_job "$figure_two_job" "figure_2_${global_seed}" "$DRY_RUN" "$figure2_resolved_deps")

  done
  done

done
done
done
done

# ============================ dump plan ============================
plan_dump
