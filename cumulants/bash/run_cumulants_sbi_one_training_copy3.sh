#!/bin/bash

RUN_LINEARISED=true # Linearised is glitching for now...
RUN_FROZEN=true
ONLY_RUN_FIGURES=false # Only run figure_one.py jobs
N_SEEDS=100
N_PARALLEL=50
N_LINEAR_SIMS=10000
N_GB=8
N_CPU=8

NON_LINEARISED_SEED=0 # Repeat this for linearised...  NOTE: add this to formatting for sbatch job names!
N_LINEAR_SEEDS=10

RUN_LINEARISED_ONLY=false # E.g. run only linearised experiments

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

            # Skip linearisation/pre-train experiments
            if [[ "$LINEARISED_FLAG" == "--linearised" && "$PRETRAIN_FLAG" == "--pre-train" ]]; then
                continue
            fi

            # Skip linearisation if not requested
            if [[ "$RUN_LINEARISED" == false && "$LINEARISED_FLAG" == "--linearised" ]]; then
                continue
            fi

            # Skip non-linearisation if not requested (if e.g. running only repeated linearised experiments)
            if [[ "$RUN_LINEARISED_ONLY" == true && "$LINEARISED_FLAG" == "--no-linearised" ]]; then
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

                    echo ">>Running redshift loop with cumulants=$order_idx_args, pretrain=$PRETRAIN_FLAG, linearised=$LINEARISED_FLAG"

                    # SBI job IDs that must all run for cumulants_multi_z.py to run for bulk/tails, linearised/no-linearised, freeze/no-freeze
                    sbi_job_ids=()
                    for z in 0.0 0.5 1.0; do

                        # Keep non-linearised seed fixed, increment seed for repeated linearised
                        if [ "$LINEARISED_FLAG" == "--linearised" ]; then # Label runs
                            for ((_LINEARISED_SEED = 0; _LINEARISED_SEED < N_LINEAR_SEEDS; x++)); do
                                _SEED=_LINEARISED_SEED
                                cmd1="python cumulants_sbi.py \
--seed $NON_LINEARISED_SEED \
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

                                job_script=$(
                                    cat <<END
#!/bin/bash
#SBATCH --job-name=sbi_${bt_flag}_${l_flag}_${f_flag}_z${z}
#SBATCH --output=$OUT_DIR/sbi_${bt_flag}_${l_flag}_${f_flag}_z${z}_fixed.out
#SBATCH --error=$OUT_DIR/sbi_${bt_flag}_${l_flag}_${f_flag}_z${z}_fixed.err
#SBATCH --partition=cluster
#SBATCH --time=06:00:00
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=end,fail

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

echo "Running sbi.py with seed $NON_LINEARISED_SEED and redshift $z"
$cmd1
END
                                )

                            done
                        else

                            cmd1="python cumulants_sbi.py \
--seed $NON_LINEARISED_SEED \
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

                            job_script=$(
                                cat <<END
#!/bin/bash
#SBATCH --job-name=sbi_${bt_flag}_${l_flag}_${f_flag}_z${z}
#SBATCH --output=$OUT_DIR/sbi_${bt_flag}_${l_flag}_${f_flag}_z${z}_fixed.out
#SBATCH --error=$OUT_DIR/sbi_${bt_flag}_${l_flag}_${f_flag}_z${z}_fixed.err
#SBATCH --partition=cluster
#SBATCH --time=06:00:00
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=end,fail

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

echo "Running sbi.py with seed $NON_LINEARISED_SEED and redshift $z"
$cmd1
END
                            )
                            sbi_job_id=$(echo "$job_script" | sbatch | awk '{print $4}')
                            sbi_job_ids+=("$sbi_job_id")
                        fi
                    done

                    sbi_deps=$(
                        IFS=":"
                        echo "${sbi_job_ids[*]}"
                    )

                    if [[ "$ONLY_RUN_FIGURES" == false ]]; then
                        if [ "$LINEARISED_FLAG" == "--linearised" ]; then # Label runs
                            for ((_LINEARISED_SEED = 0; _LINEARISED_SEED < N_LINEAR_SEEDS; x++)); do 
                                cmd2="python cumulants_multi_z.py \
--seed $_LINEARISED_SEED \
--seed_datavector \$SLURM_ARRAY_TASK_ID \
--sbi_type nle \
--compression linear \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--n_linear_sims $N_LINEAR_SIMS \
--order_idx $order_idx_args \
--bulk_or_tails $bt \
$FREEZE_FLAG"

                                # Run multi-z posterior sampling after all redshift SBI experiments are run 
                                # (separately for bulk and tails)
                                final_script=$(
                                    cat <<END
#!/bin/bash
#SBATCH --job-name=m_z_${bt_flag}_${l_flag}_${f_flag}
#SBATCH --output=$OUT_DIR/multi_z_${bt_flag}_${l_flag}_${f_flag}_%a.out
#SBATCH --error=$OUT_DIR/multi_z_${bt_flag}_${l_flag}_${f_flag}_%a.err
#SBATCH --array=0-$N_SEEDS%$N_PARALLEL
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
                            done
                        else
                            cmd2="python cumulants_multi_z.py \
--seed $NON_LINEARISED_SEED \
--seed_datavector \$SLURM_ARRAY_TASK_ID \
--sbi_type nle \
--compression linear \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--n_linear_sims $N_LINEAR_SIMS \
--order_idx $order_idx_args \
--bulk_or_tails $bt \
$FREEZE_FLAG"

                            # Run multi-z posterior sampling after all redshift SBI experiments are run 
                            # (separately for bulk and tails)
                            final_script=$(
                                cat <<END
#!/bin/bash
#SBATCH --job-name=m_z_${bt_flag}_${l_flag}_${f_flag}
#SBATCH --output=$OUT_DIR/multi_z_${bt_flag}_${l_flag}_${f_flag}_%a.out
#SBATCH --error=$OUT_DIR/multi_z_${bt_flag}_${l_flag}_${f_flag}_%a.err
#SBATCH --array=0-$N_SEEDS%$N_PARALLEL
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
                    fi

                    if [ "$LINEARISED_FLAG" == "--linearised" ]; then # Label runs
                        for ((_LINEARISED_SEED = 0; _LINEARISED_SEED < N_LINEAR_SEEDS; x++)); do 
                            figure_cmd="python figure_one.py \
--seed $_LINEARISED_SEED \
--seed_datavector \$SLURM_ARRAY_TASK_ID \
--sbi_type nle \
--compression linear \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--order_idx $order_idx_args \
$FREEZE_FLAG"

                            # Need to run this after both bulk and tails multi_z samplings
                            figure_job=$(
                                cat <<END
#!/bin/bash
#SBATCH --job-name=figure_one_linear_${_LINEARISED_SEED}
#SBATCH --output=$OUT_DIR/figure_one_${l_flag}_${f_flag}_%a.out
#SBATCH --error=$OUT_DIR/figure_one_${l_flag}_${f_flag}_%a.err
#SBATCH --array=0-$N_SEEDS%$N_PARALLEL
#SBATCH --partition=cluster
#SBATCH --time=02:00:00
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail
#SBATCH --dependency=afterok:$sbi_deps:$multi_z_deps

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

echo "Running final figure script"
$figure_cmd
END
                            )
                            echo "$figure_job" | sbatch
                        done
                    else
                        figure_cmd="python figure_one.py \
--seed $NON_LINEARISED_SEED \
--seed_datavector \$SLURM_ARRAY_TASK_ID \
--sbi_type nle \
--compression linear \
$LINEARISED_FLAG \
$PRETRAIN_FLAG \
--order_idx $order_idx_args \
$FREEZE_FLAG"

                        # Need to run this after both bulk and tails multi_z samplings
                        figure_job=$(
                            cat <<END
#!/bin/bash
#SBATCH --job-name=figure_one_nonlinear_${NON_LINEARISED_SEED}
#SBATCH --output=$OUT_DIR/figure_one_${l_flag}_${f_flag}_%a.out
#SBATCH --error=$OUT_DIR/figure_one_${l_flag}_${f_flag}_%a.err
#SBATCH --array=0-$N_SEEDS%$N_PARALLEL
#SBATCH --partition=cluster
#SBATCH --time=02:00:00
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail
#SBATCH --dependency=afterok:$sbi_deps:$multi_z_deps

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

echo "Running final figure script"
$figure_cmd
END
                        )
                        echo "$figure_job" | sbatch
                    fi
                done
            done
        done
    done
done
