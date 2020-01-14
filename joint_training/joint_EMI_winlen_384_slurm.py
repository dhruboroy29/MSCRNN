import sys
import os

sys.path.append('../')

######################### ONLY MODIFY THESE VALUES #########################
# Number of splits of hyperparam file
winlen = 384

num_splits='24'

# Base path of data
prefix = 'jointEMI_3class_winlen_' + str(winlen)

base='/scratch/dr2915/Bumblebee/bb_3class_winlen_' + str(winlen) + '_winindex_all/3class_48_16'

# Batch system
bat_sys='slurm'

# Human and nonhuman folders
#hum_fold='austere_404_human'
#nhum_fold='Bike_485_radial'

# Running time
walltime='1-0'

######################### KEEP THE REST INTACT #########################
# Folder where jobs are saved
jobfolder = '../'+ bat_sys +'_hpc/'

#Init args
init_argv=sys.argv

# Enter hpc_scripts folder
os.chdir('../hpc_scripts')

# Prepare data
#print('###### Scripts/processing_data #####')
#sys.argv=init_argv+['-type', prefix, '-base', base]
#import Scripts.create_train_val_test_split

# Generate gridsearch
print('###### hpc_scripts/gridsearch #####')
sys.argv=init_argv+['-bat', bat_sys, '-type', prefix, '-base', base, '-ots', str(winlen)]
import hpc_scripts.gridsearch_0_jointEMI

# Split hyperparam file
print('###### hpc_scripts/split_hyp_wrapper #####')
sys.argv=init_argv+[jobfolder+prefix+'.sh',num_splits]
import hpc_scripts.split_hyp_wrapper_1

# Create batch job
print('###### hpc_scripts/create_batch_wrapper #####')
sys.argv=init_argv+[jobfolder+prefix+'_',walltime,bat_sys]
import hpc_scripts.create_batch_wrapper_2

# Submit
print("\nNow submit " + bat_sys + "_hpc/3_SUBMIT_"+prefix+"_jobs.sh on server")