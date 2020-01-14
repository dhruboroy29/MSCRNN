#!/usr/bin/env bash

if [ $# -ne 1 ]
then
    echo "Usage: $0 <infolder/search_prefix>" > /dev/stderr
    exit 1
fi

dir=`dirname $1`/recall
file=`basename $1`

mkdir -p $dir

outfile="$dir/recall_$file.txt"
echo $outfile

echo "Saving output to: "$outfile
ls -1v $1*.log | xargs -d '\n' grep -E -A 20 -B 10 "Round:  [0-9], Validation accuracy:" | grep Recall\
    | awk -F'=' '{print $2}' | awk -F'[.]log[-]' '{print $1$2}' | awk -F'|' '{print $1"\t"$3"\t"$4}' | tee $outfile