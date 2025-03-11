#!/bin/bash
#SBATCH --job-name=cumulants_posteriors
#SBATCH --output=/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/%A/cumulants_exps_%a.out
#SBATCH --error=/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/%A/cumulants_exps_%a.err
#SBATCH --array=0,1%2
#SBATCH --time=02:00:00
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

        echo "Running seed=$id, redshift $z, cumulants=$order_idx_args..."

        # Sample multi-z posteriors with linear datavectors
        echo "Sampling multi-z with linear datavectors..."
        python cumulants_multi_z.py --seed $id --sbi_type "nle" --linearised --order_idx $order_idx_args --compression "linear"
        echo "Sampling multi-z with linear datavectors... Completed."

        # Sample multi-z posteriors with non-linear datavectors
        echo "Sampling multi-z with non-linear datavectors..."
        python cumulants_multi_z.py --seed $id --sbi_type "nle" --no-linearised --order_idx $order_idx_args --compression "linear"
        echo "Sampling multi-z with non-linear datavectors... Completed."

        # Sample multi-z posteriors with non-linear datavectors (pre-train)
        echo "Sampling multi-z with non-linear datavectors..."
        python cumulants_multi_z.py --seed $id --sbi_type "nle" --no-linearised --order_idx $order_idx_args --compression "linear" --pre-train
        echo "Sampling multi-z with non-linear datavectors... Completed."

        current_time=$(date +"%H:%M:%S")
        echo "The current time is: $current_time"
    done
done

end=`date +%s`

runtime=$((end-start))

echo "Time for index:" "${id}" "was" "${runtime}" "seconds"