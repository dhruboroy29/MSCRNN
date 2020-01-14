#!/usr/bin/env bash
########################################################
# Check number of args
########################################################
if [ $# -ne 2 ]
then
    echo "Usage: $0 <filename> <num_splits>" > /dev/stderr
    exit 1
fi

########################################################
# Extract file name and extension
########################################################
output_prefix="${1%.*}_"
ext="_spl.${1##*.}"

# Split file (all but the first line (outname=`echo $0 | sed "s/.sh/.out/g"`))
num_lines=`cat $1 | wc -l`
nl_per_subfile=`expr $num_lines / $2`

tail -n +2 $1 | split -l $nl_per_subfile -a ${#2} --numeric-suffixes=1 --additional-suffix=$ext - $output_prefix

# Append first line of original file (outname=`echo $0 | sed "s/.sh/.out/g"`) to each split file
for file in $output_prefix*
do
	head -n 1 $1 > tmp
	cat $file >> tmp
	mv -f tmp $file
done
