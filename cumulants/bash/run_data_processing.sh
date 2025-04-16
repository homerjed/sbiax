#!/bin/bash
#SBATCH --job-name=cumulants_data
#SBATCH --output=/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/%A/cumulants_exps_%a.out
#SBATCH --error=/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/%A/cumulants_exps_%a.err
#SBATCH --array=0
#SBATCH --time=08:00:00
#SBATCH --partition=cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
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

echo "Getting cumulants data..."
python get_cumulants_data.py 
echo "Getting cumulants data... complete."

end=`date +%s`

runtime=$((end-start))

echo "Time for index:" "${id}" "was" "${runtime}" "seconds"