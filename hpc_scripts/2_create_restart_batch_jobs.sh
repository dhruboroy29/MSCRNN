#!/usr/bin/env bash

#!/usr/bin/env bash
########################################################
# Create SLURM batch jobs
########################################################
create_batch_slurm()
{
    split_files=`(ls $1*_restart_spl.sh)`
    for file in $split_files
    do
        echo "sbatch -t $2 --export=filename=$file batch_job.sbatch"
        echo sleep 1
    done
}

########################################################
# Create PBS batch jobs
########################################################
create_batch_pbs()
{
    split_files=`(ls $1*_restart_spl.sh)`

    for file in $split_files
    do
        echo "qsub -V -l walltime=$2 -v filename=$file batch_job.pbs"
        echo sleep 1
    done
}

########################################################
# Create SLURM leftover jobs
########################################################
create_restart_jobs_slurm()
{
    dir=$(dirname "$1")
    out_files=(`find $dir -name "*spl.out"`)

    for file in ${out_files[@]}; do
	wc_op=(`wc -l $file`)
	left=$((12-${wc_op[0]}))
	fname=${wc_op[1]}
	if [ $left -gt 0 ]
	then
            sh_file=$(echo $(basename "$fname") | sed -e 's/spl.out/spl.sh/')
            restart_file=$dir/$(echo $(basename "$fname") | sed -e 's/spl.out/restart_spl.sh/')

            echo 'outname=`echo $0 | sed "s/.sh/.out/g"`' > ${restart_file}
            tail -n $left ${sh_file} >> ${restart_file}
	fi
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

create_restart_jobs_slurm $1
submit_file=$(dirname "$1")/3_SUBMIT_RESTART_$(basename "$1")jobs.sh

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