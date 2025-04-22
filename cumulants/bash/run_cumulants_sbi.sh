#!/bin/bash
#SBATCH --job-name=cumulants_exps
#SBATCH --output=/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/cumulants_sbi/%A/cumulants_exps_%a.out
#SBATCH --error=/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/cumulants_sbi/%A/cumulants_exps_%a.err
#SBATCH --array=0-10%10
#SBATCH --time=48:00:00
#SBATCH --partition=cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --mail-user=jed.homer@physik.lmu.de
#SBATCH --mail-type=begin,end,fail

current_time=$(date +"%H:%M:%S")
echo "The current time is: $current_time"

id=$SLURM_ARRAY_TASK_ID

start=`date +%s`

echo "Currently running array index: " "$id"

cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/

source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

for freeze in true false; do
    freeze_flag=""
    if [ "$freeze" = true ]; then
        freeze_flag="--freeze-parameters"
    else
        freeze_flag="--no-freeze-parameters"
    fi
    for i in {0..2}; do
        for z in 0.0 0.5 1.0; do
            order_idx_args=$(seq -s " " 0 $i)

            echo ">>Running seed=$id, redshift $z, cumulants=$order_idx_args..."

            echo ">>Running bulk linearised scripts..."
            python cumulants_sbi.py \
                --seed "$id" \
                --sbi_type "nle" \
                --compression "linear" \
                --linearised \
                --n_linear_sims 10000 \
                --order_idx $order_idx_args \
                --redshift $z \
                --no-use-tqdm \
                --bulk_or_tails "bulk" \
                "$freeze_flag"
            echo ">>Running bulk linearised scripts... Complete."

            echo ">>Running tails linearised scripts..."
            python cumulants_sbi.py \
                --seed "$id" \
                --sbi_type "nle" \
                --compression "linear" \
                --linearised \
                --n_linear_sims 10000 \
                --order_idx $order_idx_args \
                --redshift $z \
                --no-use-tqdm \
                --bulk_or_tails "tails" \
                "$freeze_flag"
            echo ">>Running tails linearised scripts... Complete."

            # echo ">>Running linearised (nn) scripts..."
            # python cumulants_sbi.py \
            #     --seed "$id" \
            #     --sbi_type "nle" \
            #     --compression "nn" \
            #     --linearised \
            #     --order_idx $order_idx_args \
            #     --redshift $z \
            #     --no-use-tqdm
            # echo ">>Running linearised (nn) scripts... Complete."

            # echo ">>Running non-linearised scripts..."
            # python cumulants_sbi.py \
            #     --seed "$id" \
            #     --sbi_type "nle" \
            #     --compression "linear" \
            #     --no-linearised \
            #     --order_idx $order_idx_args \
            #     --redshift $z \
            #     --no-use-tqdm
            # echo ">>Running non-linearised scripts... Complete."

            # echo ">>Running non-linearised (nn) scripts..."
            # python cumulants_sbi.py \
            #     --seed "$id" \
            #     --sbi_type "nle" \
            #     --compression "nn" \
            #     --no-linearised \
            #     --order_idx $order_idx_args \
            #     --redshift $z \
            #     --no-use-tqdm
            # echo ">>Running non-linearised (nn) scripts... Complete."

            echo ">>Running bulk pre-train non-linearised scripts..."
            python cumulants_sbi.py \
                --seed "$id" \
                --sbi_type "nle" \
                --compression "linear" \
                --no-linearised \
                --pre-train \
                --n_linear_sims 10000 \
                --order_idx $order_idx_args \
                --redshift $z \
                --no-use-tqdm \
                --bulk_or_tails "bulk" \
                "$freeze_flag"
            echo ">>Running bulk pre-train non-linearised scripts... Completed."

            echo ">>Running tails pre-train non-linearised scripts..."
            python cumulants_sbi.py \
                --seed "$id" \
                --sbi_type "nle" \
                --compression "linear" \
                --no-linearised \
                --pre-train \
                --n_linear_sims 10000 \
                --order_idx $order_idx_args \
                --redshift $z \
                --no-use-tqdm \
                --bulk_or_tails "tails" \
                "$freeze_flag"
            echo ">>Running tails pre-train non-linearised scripts... Completed."

            current_time=$(date +"%H:%M:%S")
            echo "The current time is: $current_time"
        done

        # Multi-z posterior sampling
            
        # Bulk, linearised
        echo ">>Sampling bulk multi-z with linearised datavectors..."
        python cumulants_multi_z.py \
            --seed "$id" \
            --sbi_type "nle" \
            --linearised \
            --order_idx $order_idx_args \
            --compression "linear" \
            --bulk_or_tails "bulk" \
            "$freeze_flag"
        echo ">>Sampling bulk multi-z with linearised datavectors... Completed."

        # Bulk, non-linearised
        echo ">>Sampling bulk multi-z with non-linearised datavectors..."
        python cumulants_multi_z.py \
            --seed "$id" \
            --sbi_type "nle" \
            --compression "linear" \
            --no-linearised \
            --pre-train \
            --n_linear_sims 10000 \
            --order_idx $order_idx_args \
            --bulk_or_tails "bulk" \
            "$freeze_flag"
        echo ">>Sampling bulk multi-z with non-linearised datavectors... Completed."

        # Tails, linearised
        echo ">>Sampling tails multi-z with linearised datavectors..."
        python cumulants_multi_z.py \
            --seed "$id" \
            --sbi_type "nle" \
            --compression "linear" \
            --linearised \
            --order_idx $order_idx_args \
            --bulk_or_tails "tails" \
            "$freeze_flag"
        echo ">>Sampling tails multi-z with linearised datavectors... Completed."

        # Tails, non-linearised
        echo ">>Sampling tails multi-z with non-linearised datavectors..."
        python cumulants_multi_z.py \
            --seed "$id" \
            --sbi_type "nle" \
            --compression "linear" \
            --no-linearised \
            --pre-train \
            --n_linear_sims 10000 \
            --order_idx $order_idx_args \
            --bulk_or_tails "tails" \
            "$freeze_flag"
        echo ">>Sampling tails multi-z with non-linearised datavectors... Completed."

        # Plot figure one

        echo ">>Plotting figure one..."
        python figure_one.py \
            --seed "$id" \
            --sbi_type "nle" \
            --compression "linear" \
            --no-pre-train \
            --order_idx $order_idx_args \
            "$freeze_flag"
        echo ">>Plotting figure one... Completed."

        python figure_one.py \
            --seed "$id" \
            --sbi_type "nle" \
            --compression "linear" \
            --pre-train \
            --order_idx $order_idx_args \
            "$freeze_flag"
        echo ">>Plotting figure one... Completed."

    done
done

end=`date +%s`
runtime=$((end-start))
echo "Time for index:" "$id" "was" "$runtime" "seconds"