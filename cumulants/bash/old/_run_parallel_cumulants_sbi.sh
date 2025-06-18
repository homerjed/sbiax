#!/bin/bash

RUN_LINEARIZED=false
FREEZE_FLAG="--no-freeze-parameters"
N_SEEDS=0
N_PARALLEL=1

TIMESTAMP=$(date +'%m%d_%H%M')
OUT_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/cumulants_sbi/$TIMESTAMP"
mkdir -p "$OUT_DIR"

all_job_ids=()

# Test datavectors [variance,] and [variance, skewness, kurtosis]
order_idxs=(
    "0"
    "0 1 2"
)

# Loop scripts for bulk and tails datavectors
for bt in "bulk" "tails"; do

    if [ "$bt" == "bulk" ]; then
        bt_flag="b"
    else
        bt_flag="t"
    fi

    # Loop over moments idx (only all cumulants at once for now)
    for order_idx_args in "${order_idxs[@]}"; do

        echo ">>Running redshift loop with cumulants=$order_idx_args..."

        # Loop over redshifts, submitting N jobs for all seeds
        job_ids=()
        for z in 0.0 0.5 1.0; do
            cmd1="python cumulants_sbi.py \
--seed \$SLURM_ARRAY_TASK_ID \
--sbi_type nle \
--compression linear \
--no-linearised \
--pre-train \
--n_linear_sims 10000 \
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
#SBATCH --mem=12G
#SBATCH --cpus-per-task=8
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
--no-linearised \
--pre-train \
--n_linear_sims 10000 \
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
#SBATCH --time=24:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=8
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
    done
done

# Submit figure_one.py as final dependent job
all_deps=$(IFS=":"; echo "${all_job_ids[*]}")

figure_cmd="python figure_one.py \
--seed \$SLURM_ARRAY_TASK_ID \
--sbi_type nle \
--compression linear \
--no-linearised \
--no-pre-train \
--order_idx 0 1 2 \
$FREEZE_FLAG"

figure_job=$(cat <<END
#!/bin/bash
#SBATCH --job-name=figure_one
#SBATCH --output=$OUT_DIR/figure_one_%a.out
#SBATCH --error=$OUT_DIR/figure_one_%a.err
#SBATCH --array=0-$N_SEEDS%$N_PARALLEL
#SBATCH --partition=cluster
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail
#SBATCH --dependency=afterok:$all_deps

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

echo "Running final figure script"
$figure_cmd
END
)

echo "$figure_job" | sbatch


    # for i in {0..2}; do
        # order_idx_args=$(seq -s " " 0 $i)