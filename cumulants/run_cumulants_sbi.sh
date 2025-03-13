#!/bin/bash
#SBATCH --job-name=cumulants_exps
#SBATCH --output=/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/%A/cumulants_exps_%a.out
#SBATCH --error=/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/%A/cumulants_exps_%a.err
#SBATCH --array=0-49%5
#SBATCH --time=48:00:00
#SBATCH --partition=cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail

current_time=$(date +"%H:%M:%S")
echo "The current time is: $current_time"

# Each process takes its ID to use for the seeds
id=$SLURM_ARRAY_TASK_ID

start=`date +%s`

echo "Currently running array index: " "${id}"

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/

# Activate env
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

# Iterate over variance, skewness & kurtosis args
for i in {0..2}; do
    # Iterate over redshift
    for z in 0.0 0.5 1.0; do
        order_idx_args=$(seq -s " " 0 $i)  # Generate "0", "0 1", "0 1 2" 

        echo ">>Running seed=$id, redshift $z, cumulants=$order_idx_args..."

        # @@@@ Per-redshift posterior fitting @@@@

        # Linear model
        echo ">>Running linear scripts..."
        python cumulants_sbi.py --seed $id --sbi_type "nle" --compression "linear" --linearised --n_linear_sims 10000 --order_idx $order_idx_args --redshift $z --no-use-tqdm
        echo ">>Running linear scripts... Complete."

        # Linear model, NN compression
        # echo ">>Running linear (nn) scripts..."
        # python cumulants_sbi.py --seed $id --sbi_type "nle" --compression "nn" --linearised --order_idx $order_idx_args --redshift $z --no-use-tqdm
        # echo ">>Running linear (nn) scripts... Complete."

        # Non-linear model, linear compression
        # echo ">>Running non-linear scripts..."
        # python cumulants_sbi.py --seed $id --sbi_type "nle" --compression "linear" --no-linearised --order_idx $order_idx_args --redshift $z --no-use-tqdm
        # echo ">>Running non-linear scripts... Complete."

        # Non-linear model, NN compression
        # echo ">>Running non-linear (nn) scripts..."
        # python cumulants_sbi.py --seed $id --sbi_type "nle" --compression "nn" --no-linearised --order_idx $order_idx_args --redshift $z --no-use-tqdm
        # echo ">>Running non-linear (nn) scripts... Complete."

        # Non-linear model (pre-train)
        echo ">>Running pre-train non-linear scripts..."
        python cumulants_sbi.py --seed $id --sbi_type "nle" --compression "linear" --no-linearised --pre-train --n_linear_sims 10000 --order_idx $order_idx_args --redshift $z --no-use-tqdm
        echo ">>Running pre-train non-linear scripts... Completed."

        # Non-linear model, NN compression (pre-train)
        # echo ">>Running pre-train non-linear scripts..."
        # python cumulants_sbi.py --seed $id --sbi_type "nle" --compression "nn" --no-linearised --pre-train --order_idx $order_idx_args --redshift $z --no-use-tqdm
        # echo ">>Running pre-train non-linear scripts... Completed."

        current_time=$(date +"%H:%M:%S")
        echo "The current time is: $current_time"
    done

    # @@@@ Multi-redshift posterior sampling @@@@
    # > Done after all redshifts are executed for this order_idx cumulants combination

    # Sample multi-z posteriors with linear datavectors
    echo ">>Sampling multi-z with linear datavectors..."
    python cumulants_multi_z.py --seed $id --sbi_type "nle" --linearised --order_idx $order_idx_args --compression "linear"
    echo ">>Sampling multi-z with linear datavectors... Completed."

    # Sample multi-z posteriors with non-linear datavectors
    # echo "Sampling multi-z with non-linear datavectors..."
    # python cumulants_multi_z.py --seed $id --sbi_type "nle" --no-linearised  --order_idx $order_idx_args --compression "linear"
    # echo "Sampling multi-z with non-linear datavectors... Completed."

    # Sample multi-z posteriors with non-linear datavectors (pre-train)
    echo ">>Sampling multi-z with non-linear datavectors..."
    python cumulants_multi_z.py --seed $id --sbi_type "nle" --compression "linear" --no-linearised --pre-train --n_linear_sims 10000 --order_idx $order_idx_args
    echo ">>Sampling multi-z with non-linear datavectors... Completed."
done

end=`date +%s`

runtime=$((end-start))

echo "Time for index:" "${id}" "was" "${runtime}" "seconds"