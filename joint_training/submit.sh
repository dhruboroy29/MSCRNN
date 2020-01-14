#!/usr/bin/env bash

list_files=(
            3_SUBMIT_jointEMI_3class_winlen_256_jobs.sh
            3_SUBMIT_jointEMI_3class_winlen_384_jobs.sh
            )

cd ../slurm_hpc

# Submit jobs
for l in ${list_files[@]}; do
    sh $l
    sleep 60
done