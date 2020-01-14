#!/usr/bin/env bash
########################################################
# Create SLURM batch jobs
########################################################
create_batch_slurm()
{
    split_files=`(ls $1*_spl.sh)`
    for file in $split_files
    do
        echo "sbatch -t $2 --export=filename=$file batch_job.sbatch"
        echo sleep 60
    done
}

########################################################
# Create PBS batch jobs
########################################################
create_batch_pbs()
{
    split_files=`(ls $1*_spl.sh)`

    for file in $split_files
    do
        echo "qsub -V -l walltime=$2 -v filename=$file batch_job.pbs"
        echo sleep 1
    done
}

########################################################
# Check number of args
########################################################
if [ $# -ne 3 ]
then
    echo "Usage: $0 <split_file_prefix> <job_time_limit> <hpc (pbs/slurm)>" > /dev/stderr
    exit 1
fi

submit_file=$(dirname "$1")/3_SUBMIT_$(basename "$1")jobs.sh

if [ $3 = 'pbs' ]
then
    create_batch_pbs $1 $2 > $submit_file
elif [ $3 = 'slurm' ]
then
    create_batch_slurm $1 $2 > $submit_file
else
echo "ERROR: hpc can only be pbs/slurm" > /dev/stderr
    exit 1
fi