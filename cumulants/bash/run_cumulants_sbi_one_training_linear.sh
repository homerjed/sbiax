#!/bin/bash

RESULTS_DIR="${1:-"results/"}"
SINGLE_RUN="${2:-false}" # Run all experiments once for a single figure one
ONLY_RUN_FIGURES="${3:-false}" # Only run figure_one.py jobs

RUN_LINEARISED=true # Linearised is glitching for now...
RUN_FROZEN=false
RUN_NONLINEAR=true

if [[ "$SINGLE_RUN" == "true" ]]; then
    echo "SINGLE RUN."
    N_SEEDS=0
    START_SEED=0
    N_SEEDS_GLOBAL=1 # Number of repeated training
    END_SEED=$(( $START_SEED + $N_SEEDS ))
    N_PARALLEL=1

    RUN_FROZEN=false
else
    echo "MULTIPLE SEEDS RUN."
    N_SEEDS=200
    START_SEED=0
    N_SEEDS_GLOBAL=10 # Number of repeated training
    END_SEED=$(( $START_SEED + $N_SEEDS ))
    N_PARALLEL=50
fi

N_GB=8
N_CPU=8

N_DATAVECTORS=10
N_LINEAR_SIMS=2000

TIMESTAMP=$(date +'%m%d_%H%M')
OUT_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/cumulants_sbi/$TIMESTAMP"
mkdir -p "$OUT_DIR"

order_idxs=(
    # "0"
    "0 1 2"
)

# Repeat training / posterior sampling for 10 seeds with linearised experiemnts

# JOB_ARRAY_STR="0-100,200-500"
# JOB_ARRAY_STR="0-200" # 200 posteriors sampled for each of the 10 seeds
JOB_ARRAY_STR="$START_SEED-$END_SEED%$N_PARALLEL"

# Empty datasets dir, recalculate them only once each
DATASETS_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/quijote_data/datasets/"
rm -rf "${DATASETS_DIR:?}/"*
echo "Emptied datasets directory: $DATASETS_DIR"

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

                for order_idx_args in "${order_idxs[@]}"; do

                    echo ">>Running redshift loop with bulk/tails=$bt, cumulants=$order_idx_args, pretrain=$PRETRAIN_FLAG, linearised=$LINEARISED_FLAG, freeze=$FREEZE_FLAG"

                    # SBI job IDs that must all run for cumulants_multi_z.py to run for bulk/tails, linearised/no-linearised, freeze/no-freeze
                    sbi_job_ids=()
                    for z in 0.0 0.5 1.0; do
                        cmd1="RESULTS_DIR=$RESULTS_DIR python cumulants_sbi.py \
--seed $global_seed \
--compression linear \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--n_linear_sims $N_LINEAR_SIMS \
--order_idx "$order_idx_args" \
--redshift $z \
--no-use-tqdm \
--bulk_or_tails $bt \
$FREEZE_FLAG"

                        job_script=$(
                            cat <<END
#!/bin/bash
#SBATCH --job-name=sbi_${global_seed}_${bt_flag}_${l_flag}_${f_flag}_z${z}
#SBATCH --output=$OUT_DIR/sbi_${bt_flag}_${l_flag}_${f_flag}_z${z}_fixed_%j.out
#SBATCH --error=$OUT_DIR/sbi_${bt_flag}_${l_flag}_${f_flag}_z${z}_fixed_%j.err
#SBATCH --partition=cluster
#SBATCH --time=06:00:00
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=end,fail

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

echo "Running sbi.py with seed $global_seed and redshift $z"
$cmd1
END
                        )
                        sbi_job_id=$(echo "$job_script" | sbatch | awk '{print $4}')
                        sbi_job_ids+=("$sbi_job_id")
                    done

                    sbi_deps=$(
                        IFS=":"
                        echo "${sbi_job_ids[*]}"
                    )

                    multi_z_job_ids=()
                    if [[ "$ONLY_RUN_FIGURES" == false ]]; then
                        cmd2="RESULTS_DIR=$RESULTS_DIR python cumulants_multi_z.py \
--seed $global_seed \
--seed_datavector \$SLURM_ARRAY_TASK_ID \
--n_datavectors $N_DATAVECTORS \
--compression linear \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--n_linear_sims $N_LINEAR_SIMS \
--order_idx "$order_idx_args" \
--bulk_or_tails $bt \
$FREEZE_FLAG"

                        # Run multi-z posterior sampling after all redshift SBI experiments are run 
                        # (separately for bulk and tails)
                        final_script=$(
                            cat <<END
#!/bin/bash
#SBATCH --job-name=m_z_${global_seed}_${bt_flag}_${l_flag}_${f_flag}
#SBATCH --output=$OUT_DIR/multi_z_${bt_flag}_${l_flag}_${f_flag}_%a_%j.out
#SBATCH --error=$OUT_DIR/multi_z_${bt_flag}_${l_flag}_${f_flag}_%a_%j.err
#SBATCH --array=$JOB_ARRAY_STR
#SBATCH --partition=cluster
#SBATCH --time=08:00:00
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail
#SBATCH --dependency=afterok:$sbi_deps

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

echo "Running final multi-z script"
$cmd2
END
                        )
                        multi_z_job_id=$(echo "$final_script" | sbatch | awk '{print $4}') # FINAL SCRIPT not JOB SCRIPT (sbi)
                        multi_z_job_ids+=("$multi_z_job_id")

                        multi_z_deps=$(
                            IFS=":"
                            echo "${multi_z_job_ids[*]}"
                        )

                        echo "$final_script" | sbatch
                    fi

                    figure_one_ids=()
                    figure_cmd="RESULTS_DIR=$RESULTS_DIR python figure_one.py \
--seed $global_seed \
--seed_datavector \$SLURM_ARRAY_TASK_ID \
--n_datavectors $N_DATAVECTORS \
--n_linear_sims $N_LINEAR_SIMS \
--compression linear \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--order_idx "$order_idx_args" \
$FREEZE_FLAG"

                    # Need to run this after both bulk and tails multi_z samplings
                    figure_job=$(
                        cat <<END
#!/bin/bash
#SBATCH --job-name=figure_one
#SBATCH --output=$OUT_DIR/figure_one_${global_seed}_${l_flag}_${f_flag}_%a_%j.out
#SBATCH --error=$OUT_DIR/figure_one_${global_seed}_${l_flag}_${f_flag}_%a_%j.err
#SBATCH --array=$JOB_ARRAY_STR
#SBATCH --partition=cluster
#SBATCH --time=02:00:00
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail
#SBATCH --dependency=afterok:$sbi_deps:$multi_z_deps

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

echo "Running final figure one script"
$figure_cmd
END
                    )
                    figure_job_id=$(echo "$figure_job" | sbatch | awk '{print $4}')
                    figure_one_ids+=("$figure_job_id")

                    figure_one_deps=$(
                        IFS=":"
                        echo "${figure_one_ids[*]}"
                    )

                done
            done
        done
    done
done
done

figure_two_cmd="
N_DATAVECTOR_SEEDS=$N_SEEDS \
N_REPEATED_SBI_SEEDS=$N_SEEDS_GLOBAL \
RESULTS_DIR=$RESULTS_DIR \
python figure_two2.py
"

figure_two_job=$(
    cat <<END
#!/bin/bash
#SBATCH --job-name=figure_two
#SBATCH --output=$OUT_DIR/figure_two_${l_flag}_${f_flag}_%j.out
#SBATCH --error=$OUT_DIR/figure_two_${l_flag}_${f_flag}_%j.err
#SBATCH --partition=cluster
#SBATCH --time=04:00:00
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail
#SBATCH --dependency=afterok:$sbi_deps:$multi_z_deps:$figure_one_deps

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

echo "Running final figure two script"
$figure_two_cmd
END
)
echo "$figure_two_job" | sbatch