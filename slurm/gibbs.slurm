#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Mon Oct 02 2023 21:49:00 GMT+1100 (Australian Eastern Daylight Time)

# Partition for the job:
#SBATCH --partition=cascade

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="gibbs"

# The project ID which this job should run under:
#SBATCH --account="punim0614"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# The amount of memory in megabytes per node:
#SBATCH --mem=16384

# Use this email address:
#SBATCH --mail-user=psnguyen@student.unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-24:0:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from this directory:
cd /data/gpfs/projects/punim0614/steven/master-thesis

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

module load foss/2022a
module load R/4.2.1-bare

Rscript sparse_f_gibbs.R $1
