#!/usr/bin/env bash
########################################################
# Check number of args
########################################################
if [ $# -ne 1 ]
then
    echo "Usage: $0 <infolder/search_prefix>" > /dev/stderr
    exit 1
fi

########################################################
# Extract filename and path
########################################################
infolder=$(dirname "$1")
search_prefix=$(basename "$1")

########################################################
# List output files
########################################################
cd $infolder
outfiles=`( ls $search_prefix*spl.out )`

########################################################
# Collate outputs
########################################################
for outfile in $outfiles
do
    cat $outfile
done
