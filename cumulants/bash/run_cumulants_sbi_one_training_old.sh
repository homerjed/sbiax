#!/bin/bash

RUN_LINEARISED=true
RUN_FROZEN=false
ONLY_RUN_FIGURES=false # Only run figure_one.py jobs
N_SEEDS=5 # Number of posteriors to measure the widths of 
N_PARALLEL=5 
N_LINEAR_SIMS=10000
N_GB=8
N_CPU=8

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

            # 
            if [[ "$LINEARISED_FLAG" == "--linearised" && "$PRETRAIN_FLAG" == "--pre-train" ]]; then
                continue
            fi

            # Skip linearised if not requested
            if [[ "$RUN_LINEARISED" == false && "$LINEARISED_FLAG" == "--linearised" ]]; then
                continue
            fi

            # Skip freezing parameters if not requested
            if [[ "$RUN_FROZEN" == false && "$FREEZE_FLAG" == "--freeze-parameters" ]]; then
                continue
            fi

            all_job_ids=()
            multi_z_job_ids=()
            
            # Dictionary to store multi_z job IDs by seed and bulk/tails
            declare -A seed_multi_z_jobs

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ start bulk / tails 
            for bt in "bulk" "tails"; do

                if [ "$bt" == "bulk" ]; then
                    bt_flag="b"
                else
                    bt_flag="t"
                fi

                if [ "$LINEARISED_FLAG" == "--linearised" ]; then
                    l_flag="l"
                else
                    l_flag="nl"
                fi


                # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ start order_idx
                for order_idx_args in "${order_idxs[@]}"; do

                    echo ">>Running redshift loop with cumulants=$order_idx_args, pretrain=$PRETRAIN_FLAG, linearised=$LINEARISED_FLAG"

                    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ start z
                    # --- Run cumulants_sbi.py ONCE with fixed seed ---
                    for z in 0.0 0.5 1.0; do
                        sbi_cmd="python cumulants_sbi.py \\
    --seed $FIXED_SEED \\
    --sbi_type nle \\
    --compression linear \\
    $LINEARISED_FLAG \\
    $PRETRAIN_FLAG \\
    --n_linear_sims $N_LINEAR_SIMS \\
    --order_idx $order_idx_args \\
    --redshift $z \\
    --no-use-tqdm \\
    --bulk_or_tails $bt \\
    $FREEZE_FLAG"

                        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=init_sbi_z${z}_${bt_flag}_${l_flag}
#SBATCH --output=$OUT_DIR/init_sbi_z${z}_${bt_flag}.out
#SBATCH --error=$OUT_DIR/init_sbi_z${z}_${bt_flag}.err
#SBATCH --partition=cluster
#SBATCH --time=04:00:00
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

$sbi_cmd
EOF
                    done
                    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ end z

                    if [[ "$ONLY_RUN_FIGURES" == false ]]; then
                        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ start seed
                        for seed in $(seq 0 $N_SEEDS); do
                            job_ids=()
                            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ start z
                            for z in 0.0 0.5 1.0; do
                                cmd1="python cumulants_sbi.py \\
    --seed $FIXED_SEED \\
    --sbi_type nle \\
    --compression linear \\
    $LINEARISED_FLAG \\
    $PRETRAIN_FLAG \\
    --n_linear_sims $N_LINEAR_SIMS \\
    --order_idx $order_idx_args \\
    --redshift $z \\
    --no-use-tqdm \\
    --bulk_or_tails $bt \\
    $FREEZE_FLAG"

                                job_id=$(sbatch <<EOF | awk '{print $4}'
#!/bin/bash
#SBATCH --job-name=c_z_${z}_${bt_flag}_${l_flag}
#SBATCH --output=$OUT_DIR/sbi_${bt_flag}_z${z}_${seed}.out
#SBATCH --error=$OUT_DIR/sbi_${bt_flag}_z${z}_${seed}.err
#SBATCH --partition=cluster
#SBATCH --time=24:00:00
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

$cmd1
EOF
)
                                job_ids+=("$job_id")
                            done
                            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ end z

                            # Join job IDs with commas for dependency
                            dependencies=$(IFS=,; echo "${job_ids[*]}")

                            cmd2="python cumulants_multi_z.py \\
    --seed $FIXED_SEED \\
    --seed_datavectors $seed \\
    --sbi_type nle \\
    --compression linear \\
    $LINEARISED_FLAG \\
    $PRETRAIN_FLAG \\
    --n_linear_sims $N_LINEAR_SIMS \\
    --order_idx $order_idx_args \\
    --bulk_or_tails $bt \\
    $FREEZE_FLAG"

                            multi_z_job_id=$(sbatch --dependency=afterok:$dependencies <<EOF | awk '{print $4}'
#!/bin/bash
#SBATCH --job-name=m_z_${bt_flag}_${l_flag}_$seed
#SBATCH --output=$OUT_DIR/multi_z_${bt_flag}_$seed.out
#SBATCH --error=$OUT_DIR/multi_z_${bt_flag}_$seed.err
#SBATCH --partition=cluster
#SBATCH --time=08:00:00
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

$cmd2
EOF
)
                            # Store the multi_z job ID for this seed and bt combination
                            seed_multi_z_jobs["${seed}_${bt_flag}"]=$multi_z_job_id
                        done
                        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ end seed
                    fi

                done
                # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ end order_idx
            done
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ end bulk / tails
            
            # Now submit figure_one.py jobs after all bulk and tails jobs are done for each seed
            if [[ "$ONLY_RUN_FIGURES" == false ]]; then
                for order_idx_args in "${order_idxs[@]}"; do
                    for seed in $(seq 0 $N_SEEDS); do
                        # Get the job IDs for both bulk and tails for this seed
                        bulk_job_id=${seed_multi_z_jobs["${seed}_b"]}
                        tails_job_id=${seed_multi_z_jobs["${seed}_t"]}
                        
                        # Only proceed if we have both job IDs
                        if [[ -n "$bulk_job_id" && -n "$tails_job_id" ]]; then
                            # Make figure_one.py job depend on both bulk and tails multi_z jobs
                            figure_cmd="python figure_one.py \\
    --seed $FIXED_SEED \\
    --seed_datavectors $seed \\
    --sbi_type nle \\
    --compression linear \\
    $LINEARISED_FLAG \\
    $PRETRAIN_FLAG \\
    --order_idx $order_idx_args \\
    $FREEZE_FLAG"

                            sbatch --dependency=afterok:$bulk_job_id,afterok:$tails_job_id <<EOF
#!/bin/bash
#SBATCH --job-name=figure_one_$seed
#SBATCH --output=$OUT_DIR/figure_one_${seed}.out
#SBATCH --error=$OUT_DIR/figure_one_${seed}.err
#SBATCH --partition=cluster
#SBATCH --time=02:00:00
#SBATCH --mem=${N_GB}GB
#SBATCH --cpus-per-task=$N_CPU
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

$figure_cmd
EOF
                        fi
                    done
                done
            fi 
        done
    done
done