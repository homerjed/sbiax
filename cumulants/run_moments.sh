#!/bin/bash
#SBATCH --job-name=moments_exps
#SBATCH --output=/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/%A/moments_exps_%a.out
#SBATCH --error=/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/%A/moments_exps_%a.err
#SBATCH --array=0-4%5
#SBATCH --time=24:00:00
#SBATCH --partition=cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail

# Each process takes its ID to use for the seeds
id=$SLURM_ARRAY_TASK_ID

start=`date +%s`

echo "Currently running array index: " "${id}"

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/moments/

# Activate env
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

# Linear model
echo "Running linear scripts..."
python moments_sbi.py --seed $id --sbi_type "nle" --linearised --redshift 0.0 --no-use-tqdm
python moments_sbi.py --seed $id --sbi_type "nle" --linearised --redshift 0.5 --no-use-tqdm
python moments_sbi.py --seed $id --sbi_type "nle" --linearised --redshift 1.0 --no-use-tqdm
echo "Running linear scripts... Complete."

# Non-linear model
echo "Running non-linear scripts..."
python moments_sbi.py --seed $id --sbi_type "nle" --no-linearised --redshift 0.0 --no-use-tqdm
python moments_sbi.py --seed $id --sbi_type "nle" --no-linearised --redshift 0.5 --no-use-tqdm
python moments_sbi.py --seed $id --sbi_type "nle" --no-linearised --redshift 1.0 --no-use-tqdm
echo "Running non-linear scripts... Complete."

# Non-linear model (pre-train)
echo "Running pre-train non-linear scripts..."
python moments_sbi.py --seed $id --sbi_type "nle" --no-linearised --pre-train --redshift 0.0 --no-use-tqdm
python moments_sbi.py --seed $id --sbi_type "nle" --no-linearised --pre-train --redshift 0.5 --no-use-tqdm
python moments_sbi.py --seed $id --sbi_type "nle" --no-linearised --pre-train --redshift 1.0 --no-use-tqdm
echo "Running pre-train non-linear scripts... Completed."

# Sample multi-redshift posterior for all the jobs run above
echo "Running multi z sampling..."
python scripts/sbi/moments/multi_z.py --seed $id --sbi_type "nle" --linearised 
python scripts/sbi/moments/multi_z.py --seed $id --sbi_type "nle" --no-linearised 
python scripts/sbi/moments/multi_z.py --seed $id --sbi_type "nle" --no-linearised --pre-train 
echo "Running multi z sampling... Complete"

# # Create initial plots
# echo "Running initial plot..."
# python scripts/analysis/initial_plot.py --seed $id --sbi_type "nle"
# echo "Running initial plot... Complete"

end=`date +%s`

runtime=$((end-start))

echo "Time for index:" "${id}" "was" "${runtime}" "seconds"