#!/bin/bash

RUN_LINEARISED=true
ONLY_RUN_FIGURES=false # Only run figure_one.py jobs
N_SEEDS=100
N_PARALLEL=10
N_LINEAR_SIMS=100000
N_GB=4
N_CPU=4

TIMESTAMP=$(date +'%m%d_%H%M')
OUT_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/cumulants_sbi/$TIMESTAMP"
mkdir -p "$OUT_DIR"

order_idxs=(
    # "0"
    "0 1 2"
)

for FREEZE_FLAG in "--freeze-parameters" "--no-freeze-parameters"; do
for LINEARISED_FLAG in "--linearised" "--no-linearised"; do
for PRETRAIN_FLAG in "--pre-train" "--no-pre-train"; do

    if [[ "$LINEARISED_FLAG" == "--linearised" && "$PRETRAIN_FLAG" == "--pre-train" ]]; then
        continue
    fi

    if [[ "$RUN_LINEARISED" == false && "$LINEARISED_FLAG" == "--linearised" ]]; then
        continue
    fi

    all_job_ids=()

    for bt in "bulk" "tails"; do

        if [ "$bt" == "bulk" ]; then
            bt_flag="b"
        else
            bt_flag="t"
        fi

        for order_idx_args in "${order_idxs[@]}"; do

            echo ">>Running redshift loop with cumulants=$order_idx_args, pretrain=$PRETRAIN_FLAG, linearised=$LINEARISED_FLAG"

            if [[ "$ONLY_RUN_FIGURES" == false ]]; then
                job_ids=()
                for z in 0.0 0.5 1.0; do
                    cmd1="python cumulants_sbi.py \
--seed \$SLURM_ARRAY_TASK_ID \
--sbi_type nle \
--compression linear \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--n_linear_sims $N_LINEAR_SIMS \
--order_idx $order_idx_args \
--redshift $z \
--no-use-tqdm \
--bulk_or_tails $bt \
$FREEZE_FLAG"

                    job_script=$(cat <<END
#!/bin/bash
#SBATCH --job-name=c_z_${z}_${bt_flag}
#SBATCH --output=$OUT_DIR/%A/sbi_${bt_flag}_z${z}_%a.out
#SBATCH --error=$OUT_DIR/%A/sbi_${bt_flag}_z${z}_%a.err
#SBATCH --array=0-$N_SEEDS%$N_PARALLEL
#SBATCH --partition=cluster
#SBATCH --time=24:00:00
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

echo "Running redshift $z command"
$cmd1
END
)
                    job_id=$(echo "$job_script" | sbatch | awk '{print $4}')
                    job_ids+=("$job_id")
                    all_job_ids+=("$job_id")
                done

                deps=$(IFS=":"; echo "${job_ids[*]}")

                cmd2="python cumulants_multi_z.py \
--seed \$SLURM_ARRAY_TASK_ID \
--sbi_type nle \
--compression linear \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--n_linear_sims $N_LINEAR_SIMS \
--order_idx $order_idx_args \
--bulk_or_tails $bt \
$FREEZE_FLAG"

                final_script=$(cat <<END
#!/bin/bash
#SBATCH --job-name=m_z_${bt_flag}
#SBATCH --output=$OUT_DIR/%A/multi_z_${bt_flag}_%a.out
#SBATCH --error=$OUT_DIR/%A/multi_z_${bt_flag}_%a.err
#SBATCH --array=0-$N_SEEDS%$N_PARALLEL
#SBATCH --partition=cluster
#SBATCH --time=08:00:00
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail
#SBATCH --dependency=afterok:$deps

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

echo "Running final multi-z script"
$cmd2
END
)
                final_job_id=$(echo "$final_script" | sbatch | awk '{print $4}')
                all_job_ids+=("$final_job_id")
            fi

            # Use dummy dependency if skipping jobs
            if [[ "$ONLY_RUN_FIGURES" == true ]]; then
                all_deps=""
            else
                all_deps=$(IFS=":"; echo "${all_job_ids[*]}")
            fi

            figure_cmd="python figure_one.py \
--seed \$SLURM_ARRAY_TASK_ID \
--sbi_type nle \
--compression linear \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--order_idx $order_idx_args \
$FREEZE_FLAG"

            figure_job=$(cat <<END
#!/bin/bash
#SBATCH --job-name=figure_one
#SBATCH --output=$OUT_DIR/figure_one_%a.out
#SBATCH --error=$OUT_DIR/figure_one_%a.err
#SBATCH --array=0-$N_SEEDS%$N_PARALLEL
#SBATCH --partition=cluster
#SBATCH --time=02:00:00
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail
${all_deps:+#SBATCH --dependency=afterok:$all_deps}

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

echo "Running final figure script"
$figure_cmd
END
)
            echo "$figure_job" | sbatch

        done
    done
done
done
done