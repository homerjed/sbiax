import os
import datetime
import subprocess
import shutil
import textwrap
import tempfile

def empty_dir(path):
    for filename in os.listdir(path):
        full_path = os.path.join(path, filename)
        if os.path.isfile(full_path) or os.path.islink(full_path):
            os.remove(full_path)
        elif os.path.isdir(full_path):
            shutil.rmtree(full_path)

def get_log_dir(results_dir, *args):
    results_dir = results_dir.rstrip('/')
    return os.path.join(results_dir, *args)

# Command line args
RESULTS_DIR = "results/"
SINGLE_RUN = False # Run all experiments once for a single figure one
ONLY_RUN_FIGURES = False # Only run figure_one.py jobs

RUN_LINEARISED = True
RUN_NONLINEAR = True

RUN_FROZEN = False
USE_PLANCK = False

DEFAULT_NDE_TYPE = "CNF"
DEFAULT_N_NDES = 1

FORCE_NOISELESS_DATAVECTOR = False
USE_QUIJOTE_TAILS = False # Use tails datavectors measured, not calculated, from Quijote
FIDUCIAL_REDUCE = True # Reduce cumulants with fiducial variances
USE_SCALERS = True # Not implemented yet

DEFAULT_RESOLUTION = 1024

# Running a test single run or not
if SINGLE_RUN:
    print("SINGLE RUN.")
    N_SEEDS = 1
    START_SEED = 0
    N_SEEDS_GLOBAL = 1    # Number of repeated trainings for SBI 
    END_SEED = START_SEED + N_SEEDS - 1
    N_PARALLEL = 2

    RUN_FROZEN = False
else:
    print("MULTIPLE SEEDS RUN.")
    N_SEEDS = 20
    START_SEED = 0
    N_SEEDS_GLOBAL = 1   # Number of repeated trainings for SBI
    END_SEED = START_SEED + N_SEEDS - 1
    N_PARALLEL = 100

N_GB = 8
N_CPU = 8
JOB_TIME = "02:00:00"
MAIL_TYPE = "begin,end,fail"
JOB_ARRAY_STR = f"{START_SEED}-{END_SEED}%{N_PARALLEL}"

N_DATAVECTORS = 10        # Number of independent datavectors to sample posteriors with    
N_LINEAR_SIMS = 2000      # Number of linear simulations to use for training / pre-training

order_idxs = (
    # "0"
    "0 1 2",
    # [0, 1, 2],
)

scales_sets = (
    # "15.0 20.0 25.0 30.0 35.0"
    "5.0 10.0 15.0 20.0 25.0 30.0 35.0",
    # [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
)

all_redshifts = [0.0, 0.5, 1.0]

# SBATCH out directory
TIMESTAMP = datetime.datetime.now().strftime("%m%d_%H%M")
OUT_DIR = f"/project/ls-gruen/users/jed.homer/sbiaxpdf/sbatch_outs/cumulants_sbi/{TIMESTAMP}"
os.makedirs(OUT_DIR, exist_ok=True)

# Empty datasets dir, recalculate them only once each
DATASETS_DIR = "/project/ls-gruen/users/jed.homer/sbiaxpdf/quijote_data/datasets/"
empty_dir(DATASETS_DIR)
print(f"Emptied datasets directory: {DATASETS_DIR}")

# Base logging directory
BASE_LOG_DIR = f"/project/ls-gruen/users/jed.homer/sbiaxpdf/{RESULTS_DIR}/logs/"
os.makedirs(BASE_LOG_DIR, exist_ok=True)

LOG_LEVEL = "DEBUG"

if USE_PLANCK: 
    USE_PLANCK_FLAG="--use-planck"
else:
    USE_PLANCK_FLAG="--no-use-planck"

raw_data_dir = "/project/ls-gruen/users/jed.homer/sbiaxpdf/quijote_data/raw/"

# Run cumulants-from-quijote if using high resolution
data_deps = []

data_dependency_line = "" # Empty by default, overwrite with cumulants + pdfs jobs if required

if USE_QUIJOTE_TAILS:
    if all(
        os.path.exists(os.path.join(raw_data_dir, _))
        for _ in [
            f"ALL_FIDUCIAL_PDFS_resolution={DEFAULT_RESOLUTION}.npy",
            f"ALL_LATIN_PDFS_resolution={DEFAULT_RESOLUTION}.npy",
            f"pdfs_derivatives_plus_minus_resolution={DEFAULT_RESOLUTION}.npy"
        ]
    ):
        print("Loaded cumulants data")

    else:
        cumulants_data_log_dir=get_log_dir(BASE_LOG_DIR, "cumulants_data")

        cumulants_cmd=f"""\
        #!/bin/bash
        #SBATCH --job-name=cumulants_data
        #SBATCH --output={OUT_DIR}/cumulants_data.out
        #SBATCH --error={OUT_DIR}/cumulants_data.err
        #SBATCH --partition=cluster
        #SBATCH --time={JOB_TIME}
        #SBATCH --mem={N_GB}GB
        #SBATCH --cpus-per-task={N_CPU}
        #SBATCH --mail-user=jed.homer@physik.lmu.de
        #SBATCH --mail-type={MAIL_TYPE}

        cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/data/
        source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

        mkdir -p "{cumulants_data_log_dir}"

        export N_DATAVECTOR_SEEDS={N_SEEDS} 
        export N_REPEATED_SBI_SEEDS={N_SEEDS_GLOBAL} 
        export LOG_DIR={cumulants_data_log_dir}/
        export LOG_LEVEL={LOG_LEVEL}
        export RESULTS_DIR={RESULTS_DIR}
        export DEFAULT_NDE_TYPE={DEFAULT_NDE_TYPE}
        export DEFAULT_N_NDES={DEFAULT_N_NDES}
        export FORCE_NOISELESS_DATAVECTOR={FORCE_NOISELESS_DATAVECTOR}
        export USE_QUIJOTE_TAILS={USE_QUIJOTE_TAILS}
        export FIDUCIAL_REDUCE={FIDUCIAL_REDUCE}
        export DEFAULT_RESOLUTION={DEFAULT_RESOLUTION}

        echo "Running cumulants data script"

        python get_cumulants_data.py
        """

        print("Running cumulants data scripts")

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as f:
            f.write(cumulants_cmd)
            temp_script_path = f.name

            # Submit the job by piping the script into sbatch and capturing the job ID
            proc = subprocess.Popen(['sbatch', temp_script_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = proc.communicate()

            if proc.returncode != 0:
                raise RuntimeError(f"sbatch failed:\n{stderr}")

        # Extract job ID (assuming output like: "Submitted batch job 123456")
        data_job_id = stdout.strip().split()[-1]

        data_deps.append(data_job_id)

        data_dep_str = ":".join(data_deps)
        
        data_dependency_line = f"#SBATCH --dependency=afterok:{data_dep_str}"
else:
    print("Not running cumulants data scripts")

# PDFs data
if all(
    os.path.exists(os.path.join(raw_data_dir, _))
    for _ in [
        f"ALL_FIDUCIAL_PDFS_resolution={DEFAULT_RESOLUTION}.npy",
        f"ALL_LATIN_PDFS_resolution={DEFAULT_RESOLUTION}.npy",
        f"pdfs_derivatives_plus_minus_resolution={DEFAULT_RESOLUTION}.npy"
    ]
):
    print("Loaded PDF data")

else:
    # Always run PDFs
    pdfs_data_log_dir = get_log_dir(BASE_LOG_DIR, "pdfs_data")

    pdfs_cmd = f"""\
    #!/bin/bash
    #SBATCH --job-name=pdfs_data
    #SBATCH --output={OUT_DIR}/pdfs_data.out
    #SBATCH --error={OUT_DIR}/pdfs_data.err
    #SBATCH --partition=cluster
    #SBATCH --time={JOB_TIME}
    #SBATCH --mem={N_GB}GB
    #SBATCH --cpus-per-task={N_CPU}
    #SBATCH --mail-user=jed.homer@physik.lmu.de
    #SBATCH --mail-type={MAIL_TYPE}

    cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/data/
    source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

    mkdir -p "{pdfs_data_log_dir}"

    export N_DATAVECTOR_SEEDS={N_SEEDS}
    export N_REPEATED_SBI_SEEDS={N_SEEDS_GLOBAL} 
    export LOG_DIR="{pdfs_data_log_dir}/"
    export LOG_LEVEL="{LOG_LEVEL}"
    export RESULTS_DIR="{RESULTS_DIR}"
    export DEFAULT_NDE_TYPE="{DEFAULT_NDE_TYPE}"
    export DEFAULT_N_NDES="{DEFAULT_N_NDES}"
    export FORCE_NOISELESS_DATAVECTOR="{FORCE_NOISELESS_DATAVECTOR}"
    export USE_QUIJOTE_TAILS="{USE_QUIJOTE_TAILS}"
    export FIDUCIAL_REDUCE="{FIDUCIAL_REDUCE}"
    export DEFAULT_RESOLUTION="{DEFAULT_RESOLUTION}"

    echo "Running PDFs data script"

    python get_pdf_data.py
    """

    print("Running PDF data scripts")

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as f:
        f.write(pdfs_cmd)
        f.flush()
        temp_script_path = f.name

        # Submit the job by piping the script into sbatch and capturing the job ID
        proc = subprocess.Popen(['sbatch', temp_script_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed:\n{stderr}")

    data_job_id = stdout.strip().split()[-1]

    data_deps.append(data_job_id)

    data_dep_str = ":".join(data_deps)

    data_dependency_line = f"#SBATCH --dependency=afterok:{data_dep_str}"

# Run all SBI jobs
for global_seed in range(N_SEEDS_GLOBAL):

    # Record jobs for each seed; such that once SBI is run for all z, multi-z starts, then figure one
    sbi_job_ids = []
    multi_z_job_ids = []
    figure_one_job_ids = []

    for FREEZE_FLAG in ["--freeze_parameters", "--no-freeze-parameters"]:

        if FREEZE_FLAG == "--freeze-parameters":
            f_flag = "f"
        else:
            f_flag = "nf"

        if FREEZE_FLAG == "--freeze-parameters" and not RUN_FROZEN:
            continue
        if FREEZE_FLAG == "--freeze-parameters" and USE_PLANCK:
            continue
            
        for LINEARISED_FLAG in ["--linearised", "--no-linearised"]:

            if LINEARISED_FLAG == "--no-linearised" and not RUN_NONLINEAR:
                continue

            if LINEARISED_FLAG == "--linearised":
                l_flag = "l"
            if LINEARISED_FLAG == "--no-linearised":
                l_flag = "nl"

            for PRETRAIN_FLAG in ["--pre-train", "--no-pre-train"]:

                if PRETRAIN_FLAG == "--pre-train":
                    continue

                for bt in ["bulk", "tails"]:

                    if bt == "bulk":
                        bt_flag = "b"
                    if bt == "tails":
                        bt_flag = "t"

                    for scale_args in scales_sets:

                        scale_str = scale_args.replace(" ", "") # "".join(scale_args)

                        for order_idx_args in order_idxs:

                            order_idx_str = order_idx_args.replace(" ", "") #"".join(order_idx_args)

                            for z in all_redshifts:

                                print(
                                    "Running SBI job for: \n\tseed={} \n\tz={}, \n\tbulk/tails={}, \n\tcumulants={}, \n\tpre-train={}, \n\tlinearised={}, \n\tfreeze={}".format(
                                        global_seed, z, bt, order_idx_args, PRETRAIN_FLAG, LINEARISED_FLAG, FREEZE_FLAG
                                    )
                                )

                                sbi_log_dir = get_log_dir(
                                    BASE_LOG_DIR, 
                                    "sbi", 
                                    str(z), 
                                    bt, 
                                    order_idx_str, 
                                    scale_str, 
                                    LINEARISED_FLAG, 
                                    PRETRAIN_FLAG, 
                                    str(global_seed)
                                )
                                print("\n")

                                sbi_job_script = textwrap.dedent(
                                    f"""#!/bin/bash
                                        #SBATCH --job-name=sbi_{global_seed}_{bt_flag}_{l_flag}_{f_flag}_z{z}
                                        #SBATCH --output={OUT_DIR}/{bt_flag}/{l_flag}/{f_flag}/z{z}/sbi_fixed_%j.out
                                        #SBATCH --error={OUT_DIR}/{bt_flag}/{l_flag}/{f_flag}/z{z}/sbi_fixed_%j.err
                                        #SBATCH --partition=cluster
                                        #SBATCH --time={JOB_TIME}
                                        #SBATCH --mem={N_GB}GB
                                        #SBATCH --cpus-per-task={N_CPU}
                                        #SBATCH --mail-user=jed.homer@physik.lmu.de
                                        #SBATCH --mail-type={MAIL_TYPE}
                                        {data_dependency_line}

                                        cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
                                        source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

                                        mkdir -p "{sbi_log_dir}"

                                        export LOG_DIR={sbi_log_dir}
                                        export LOG_LEVEL={LOG_LEVEL}
                                        export RESULTS_DIR={RESULTS_DIR}
                                        export DEFAULT_NDE_TYPE={DEFAULT_NDE_TYPE}
                                        export DEFAULT_N_NDES={DEFAULT_N_NDES}
                                        export FORCE_NOISELESS_DATAVECTOR={FORCE_NOISELESS_DATAVECTOR}
                                        export USE_QUIJOTE_TAILS={USE_QUIJOTE_TAILS}
                                        export FIDUCIAL_REDUCE={FIDUCIAL_REDUCE}
                                        export DEFAULT_RESOLUTION={DEFAULT_RESOLUTION}

                                        python cumulants_sbi.py 
                                        --seed {global_seed} 
                                        --compression "linear"
                                        {LINEARISED_FLAG} 
                                        {PRETRAIN_FLAG} 
                                        --n_linear_sims {N_LINEAR_SIMS} 
                                        --order_idx {order_idx_args} 
                                        --scales {scale_args} 
                                        --redshift {z} 
                                        --no-use-tqdm 
                                        --bulk_or_tails "{bt}"
                                        {USE_PLANCK_FLAG} 
                                        {FREEZE_FLAG}

                                        echo "Running sbi script with seed {global_seed} and redshift {z}"
                                    """
                                )
                                print(sbi_job_script)

                                with tempfile.NamedTemporaryFile("w", suffix=".sh") as f:
                                    f.write(sbi_job_script)
                                    f.flush()  
                                    temp_script_path = f.name

                                    # Submit the job by piping the script into sbatch and capturing the job ID
                                    proc = subprocess.Popen(['sbatch', temp_script_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                                    stdout, stderr = proc.communicate()

                                    if proc.returncode != 0:
                                        raise RuntimeError(f"sbatch failed:\n{stderr}")

                                sbi_job_id = stdout.strip().split()[-1]

                                sbi_job_ids.append(sbi_job_id)

                                sbi_dep_str = ":".join(sbi_job_ids)
            
                            multi_z_log_dir = get_log_dir(
                                BASE_LOG_DIR,
                                "multi_z",
                                bt,
                                order_idx_str,
                                scale_str,
                                LINEARISED_FLAG,
                                PRETRAIN_FLAG,
                                str(global_seed)
                            )

                            print(
                                "Runnnig multi-z job for: \n\tseed={} \n\tbulk/tails={}, \n\tcumulants={}, \n\tpre-train={}, \n\tlinearised={}, \n\tfreeze={}".format(
                                    global_seed, bt, order_idx_args, PRETRAIN_FLAG, LINEARISED_FLAG, FREEZE_FLAG
                                )
                            )

                            # Run multi-z posterior sampling after all redshift SBI experiments are run 
                            # (separately for bulk and tails)
                            multi_z_script = textwrap.dedent(
                                f"""#!/bin/bash
                                    #SBATCH --job-name=m_z_{global_seed}_{bt_flag}_{l_flag}_{f_flag}
                                    #SBATCH --output={OUT_DIR}/{global_seed}/{bt_flag}/{l_flag}/{f_flag}/m_z_%a_%j.out
                                    #SBATCH --error={OUT_DIR}/{global_seed}/{bt_flag}/{l_flag}/{f_flag}/m_z_%a_%j.err
                                    #SBATCH --array={JOB_ARRAY_STR}
                                    #SBATCH --partition=cluster
                                    #SBATCH --time={JOB_TIME}
                                    #SBATCH --mem={N_GB}GB
                                    #SBATCH --cpus-per-task={N_CPU}
                                    #SBATCH --mail-user=jed.homer@physik.lmu.de
                                    #SBATCH --mail-type={MAIL_TYPE}
                                    #SBATCH --dependency=afterok:{sbi_dep_str}

                                    cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
                                    source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

                                    mkdir -p "{multi_z_log_dir}/$SLURM_ARRAY_TASK_ID"

                                    export LOG_DIR="{multi_z_log_dir}/$SLURM_ARRAY_TASK_ID"
                                    export LOG_LEVEL={LOG_LEVEL}
                                    export RESULTS_DIR={RESULTS_DIR}
                                    export DEFAULT_NDE_TYPE={DEFAULT_NDE_TYPE}
                                    export DEFAULT_N_NDES={DEFAULT_N_NDES}
                                    export FORCE_NOISELESS_DATAVECTOR={FORCE_NOISELESS_DATAVECTOR}
                                    export USE_QUIJOTE_TAILS={USE_QUIJOTE_TAILS}
                                    export FIDUCIAL_REDUCE={FIDUCIAL_REDUCE}
                                    export DEFAULT_RESOLUTION={DEFAULT_RESOLUTION}

                                    echo "Running final multi-z script"
                                    python cumulants_multi_z.py 
                                        --seed {global_seed} 
                                        --seed_datavector $SLURM_ARRAY_TASK_ID 
                                        --n_datavectors {N_DATAVECTORS} 
                                        --compression linear 
                                        {LINEARISED_FLAG} 
                                        {PRETRAIN_FLAG} 
                                        --n_linear_sims {N_LINEAR_SIMS} 
                                        --order_idx {order_idx_args} 
                                        --scales {scale_args} 
                                        --bulk_or_tails "{bt}"
                                        {USE_PLANCK_FLAG} 
                                        {FREEZE_FLAG}

                                    echo -e ">>Running Multi-z with: \nglobal_seed=$global_seed, \ndatavector_seed=$SLURM_ARRAY_TASK_ID, \nz=$z, \nbulk/tails=$bt, \ncumulants=$order_idx_args, \npretrain=$PRETRAIN_FLAG, \nlinearised=$LINEARISED_FLAG, \nfreeze=$FREEZE_FLAG"
                                """
                            )

                            with tempfile.NamedTemporaryFile("w", suffix=".sh") as f:
                                f.write(multi_z_script)
                                f.flush()  
                                temp_script_path = f.name
                            
                                # Submit the job by piping the script into sbatch and capturing the job ID
                                proc = subprocess.Popen(['sbatch', temp_script_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                                stdout, stderr = proc.communicate()

                                if proc.returncode != 0:
                                    raise RuntimeError(f"sbatch failed:\n{stderr}")

                            multi_z_job_id = stdout.strip().split()[-1]

                            multi_z_job_ids.append(multi_z_job_id)

                            multi_z_dep_str = ":".join(multi_z_job_ids)

                            # Figure one jobs
                            figure_one_log_dir = get_log_dir(
                                BASE_LOG_DIR, 
                                "figure_one",
                                bt,
                                order_idx_str,
                                scale_str,
                                LINEARISED_FLAG,
                                PRETRAIN_FLAG,
                                str(global_seed),
                            )

                            print(
                                "Runnnig figure one job for: \n\tseed={} \n\tbulk/tails={}, \n\tcumulants={}, \n\tpre-train={}, \n\tlinearised={}, \n\tfreeze={}".format(
                                    global_seed, bt, order_idx_args, PRETRAIN_FLAG, LINEARISED_FLAG, FREEZE_FLAG
                                )
                            )

                            # Need to run this after both bulk and tails multi_z samplings
                            figure_one_job = textwrap.dedent(
                                f"""#!/bin/bash
                                    #SBATCH --job-name=figure_one
                                    #SBATCH --output={OUT_DIR}/{global_seed}/{l_flag}/{f_flag}/figure_one_%a_%j.out
                                    #SBATCH --error={OUT_DIR}/{global_seed}/{l_flag}/{f_flag}/figure_one_%a_%j.err
                                    #SBATCH --array={JOB_ARRAY_STR}
                                    #SBATCH --partition=cluster
                                    #SBATCH --time={JOB_TIME}
                                    #SBATCH --mem={N_GB}GB
                                    #SBATCH --cpus-per-task={N_CPU}
                                    #SBATCH --mail-user=jed.homer@physik.lmu.de
                                    #SBATCH --mail-type={MAIL_TYPE}
                                    #SBATCH --dependency=afterok:{sbi_dep_str}:{multi_z_dep_str}

                                    cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
                                    source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

                                    mkdir -p "{figure_one_log_dir}/$SLURM_ARRAY_TASK_ID"

                                    export LOG_DIR={figure_one_log_dir}/$SLURM_ARRAY_TASK_ID
                                    export LOG_LEVEL={LOG_LEVEL}
                                    export RESULTS_DIR={RESULTS_DIR}
                                    export DEFAULT_NDE_TYPE={DEFAULT_NDE_TYPE}
                                    export DEFAULT_N_NDES={DEFAULT_N_NDES}
                                    export FORCE_NOISELESS_DATAVECTOR={FORCE_NOISELESS_DATAVECTOR}
                                    export USE_QUIJOTE_TAILS={USE_QUIJOTE_TAILS}
                                    export FIDUCIAL_REDUCE={FIDUCIAL_REDUCE}
                                    export DEFAULT_RESOLUTION={DEFAULT_RESOLUTION}

                                    python figure_one.py 
                                    --seed {global_seed} 
                                    --seed_datavector $SLURM_ARRAY_TASK_ID 
                                    --n_datavectors {N_DATAVECTORS} 
                                    --n_linear_sims {N_LINEAR_SIMS} 
                                    --compression linear 
                                    {LINEARISED_FLAG} 
                                    {PRETRAIN_FLAG} 
                                    --order_idx {order_idx_args} 
                                    --scales {scale_args} 
                                    {USE_PLANCK_FLAG} 
                                    {FREEZE_FLAG}

                                    echo -e ">>Running figure one with: \nglobal_seed=$global_seed, \ndatavector_seed=$SLURM_ARRAY_TASK_ID, \ncumulants=$order_idx_args, \npretrain=$PRETRAIN_FLAG, \nlinearised=$LINEARISED_FLAG, \nfreeze=$FREEZE_FLAG"
                                """
                            )

                            with tempfile.NamedTemporaryFile("w", suffix=".sh") as f:
                                f.write(figure_one_job)
                                f.flush()  
                                temp_script_path = f.name

                                # Submit the job by piping the script into sbatch and capturing the job ID
                                proc = subprocess.Popen(['sbatch', temp_script_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                                stdout, stderr = proc.communicate()

                                # os.system("sbatch {}".format(temp_script_path))

                                if proc.returncode != 0:
                                    raise RuntimeError(f"sbatch failed:\n{stderr}")

                            figure_one_job_id = stdout.strip().split()[-1]

                            figure_one_job_ids.append(figure_one_job_id)

                            figure_one_dep_str = ":".join(figure_one_job_ids)

    print("@" * 80)
    print("SUBMITTED JOBS FOR SEED {}".format(global_seed))

# Figure two
for FREEZE_FLAG in ["--freeze_parameters", "--no-freeze-parameters"]:

    if FREEZE_FLAG == "--freeze-parameters":
        f_flag = "f"
    else:
        f_flag = "nf"

    if FREEZE_FLAG == "--freeze-parameters" and not RUN_FROZEN:
        continue
    if FREEZE_FLAG == "--freeze-parameters" and USE_PLANCK:
        continue

    for LINEARISED_FLAG in ["--linearised", "--no-linearised"]:

        if LINEARISED_FLAG == "--no-linearised" and not RUN_NONLINEAR:
            continue

        if LINEARISED_FLAG == "--linearised":
            l_flag = "l"
        if LINEARISED_FLAG == "--no-linearised":
            l_flag = "nl"

        for PRETRAIN_FLAG in ["--pre-train", "--no-pre-train"]:

            if PRETRAIN_FLAG == "--pre-train":
                continue

            for bt in ["bulk", "tails"]:

                if bt == "bulk":
                    bt_flag = "b"
                if bt == "tails":
                    bt_flag = "t"

                for scale_args in scales_sets:

                    scale_str = scale_args.replace(" ", "") # "".join(scale_args)

                    for order_idx_args in order_idxs:

                        order_idx_str = order_idx_args.replace(" ", "") #"".join(order_idx_args)

                        print(f">>Running figure two with: \n\tcumulants={order_idx_args}, \n\tpretrain={PRETRAIN_FLAG}, \n\tlinearised={LINEARISED_FLAG}, \n\tfreeze={FREEZE_FLAG}")

                        figure_two_log_dir = get_log_dir(
                            BASE_LOG_DIR,
                            "figure_two",
                            order_idx_str,
                            scale_str
                        )

                        figure_two_job = textwrap.dedent(
                            f"""#!/bin/bash
                                #SBATCH --job-name=figure_two
                                #SBATCH --output={OUT_DIR}/{l_flag}/{f_flag}/figure_two_%j.out
                                #SBATCH --error={OUT_DIR}/{l_flag}/{f_flag}/figure_two_%j.err
                                #SBATCH --partition=cluster
                                #SBATCH --time={JOB_TIME}
                                #SBATCH --mem={N_GB}GB
                                #SBATCH --cpus-per-task={N_CPU}
                                #SBATCH --mail-user=jed.homer@physik.lmu.de
                                #SBATCH --mail-type={MAIL_TYPE}
                                #SBATCH --dependency=afterok:{sbi_dep_str}:{multi_z_dep_str}:{figure_one_dep_str}

                                cd /project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/
                                source /project/ls-gruen/users/jed.homer/sbiaxpdf/.venv/bin/activate

                                mkdir -p "{figure_two_log_dir}"

                                export N_DATAVECTOR_SEEDS={N_SEEDS}
                                export N_REPEATED_SBI_SEEDS={N_SEEDS_GLOBAL}
                                export LOG_DIR={figure_two_log_dir}/
                                export LOG_LEVEL={LOG_LEVEL}
                                export RESULTS_DIR={RESULTS_DIR}
                                export DEFAULT_NDE_TYPE={DEFAULT_NDE_TYPE}
                                export DEFAULT_N_NDES={DEFAULT_N_NDES}
                                export FORCE_NOISELESS_DATAVECTOR={FORCE_NOISELESS_DATAVECTOR}
                                export USE_QUIJOTE_TAILS={USE_QUIJOTE_TAILS}
                                export FIDUCIAL_REDUCE={FIDUCIAL_REDUCE}
                                export DEFAULT_RESOLUTION={DEFAULT_RESOLUTION}

                                echo "Running final figure two script"

                                python figure_two.py 
                                --n_datavectors {N_DATAVECTORS} 
                                --compression linear 
                                --n_linear_sims {N_LINEAR_SIMS} 
                                --order_idx {order_idx_args} 
                                --scales {scale_args} 
                                {LINEARISED_FLAG} 
                                {PRETRAIN_FLAG} 
                                {USE_PLANCK_FLAG} 
                                {FREEZE_FLAG}
                            """
                        )

                        with tempfile.NamedTemporaryFile("w", suffix=".sh") as f:
                            f.write(figure_two_job)
                            f.flush()
                            temp_script_path = f.name

                            # Submit the job by piping the script into sbatch and capturing the job ID
                            proc = subprocess.Popen(['sbatch', temp_script_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                            stdout, stderr = proc.communicate()

                            if proc.returncode != 0:
                                raise RuntimeError(f"sbatch failed:\n{stderr}")