#!/bin/bash

RESULTS_DIR="results_.../" #"${1:-"results/"}"

echo "Running with " "$RESULTS_DIR"

BASH_SCRIPTS_DIR="/project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/bash/" 

# Run single training on Quijote, sample many posteriors
bash "$BASH_SCRIPTS_DIR/run_cumulants_sbi_one_training.sh" "$RESULTS_DIR"
 
# Run multiple trainings on linearised-Quijote, sample many posteriors for each training
bash "$BASH_SCRIPTS_DIR/run_cumulants_sbi_one_training_linear.sh" "$RESULTS_DIR"

# Run Figure Two plot using all these results
figure_job=$(
    cat <<EOF
#!/bin/bash
#SBATCH --job-name=figure_two
#SBATCH --output=figure_two_%a.out
#SBATCH --error=figure_two_%a.err
#SBATCH --partition=cluster
#SBATCH --time=02:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

echo "Running final figure script"
RESULTS_DIR=$RESULTS_DIR python figure_two.py
EOF
)
echo "$figure_job" | sbatch