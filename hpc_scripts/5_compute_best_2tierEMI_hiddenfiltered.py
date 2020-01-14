#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys, os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

filename = sys.argv[1]
#outfilename = sys.argv[2]
hiddensizefilter =  sys.argv[2]

df = pd.read_table(filename, header=None,
                   names=['gN', 'uN', 'uR', 'wR', 'rnd', 'ep', 'it', 'bs', 'H', 'H2', 'k', 'total_savings', 'modelsize', 'Val_Acc', 'Acc', 'Recall0', 'Recall1', 'Recall2'])

# Only taking rows wh valid accuracy
df['Acc'].replace('', np.nan, inplace=True)
df.dropna(subset=['Acc'], inplace=True)

# Filter by hiddensize
df = df.loc[df['H'] == int(hiddensizefilter)]

# Compute average accuracy, grouping by hyperparams
#df_cv = df['Acc'].groupby([df['ggnl'], df['gunl'], df['ur'], df['wr'],
#                           df['w'], df['sp'], df['lr'], df['bs'], df['hs'], df['ot'], df['ml']]).mean().to_frame()

# Save to groupby file
#df_cv.to_csv(outfilename, sep="\t", quoting=csv.QUOTE_NONE)

# Show best hyperparams
max = df['Acc'].max()
idx = df.loc[df['Acc'].idxmax()].tolist()

print('Best Test accuracy:', str(max))
print('Corresponding params')
print("\t".join([str(i) for i in ['gN', 'uN', 'uR', 'wR', 'rnd', 'ep', 'it', 'bs', 'H', 'H2', 'k', 'total_savings', 'modelsize', 'Val_Acc', 'Acc', 'Recall0', 'Recall1', 'Recall2']]))
print("\t".join([str(i) for i in idx]))

# Create rerun string for best hyperparams
#param_str = ['-ggnl', '-gunl', '-ur', '-wr', '-w', '-sp', '-lr', '-bs', '-hs', '-ot', '-ml']
#print('Best hyperparam string')
#print(" ".join([str(item) for sublist in list(map(list,zip(param_str,idx))) for item in sublist]))

# Get corresponding line in script file
#with open(os.path.join(os.path.dirname(filename), os.path.splitext(os.path.basename(filename))[0]+'.sh'), "r") as f:
#    for t in np.arange(df['Acc'].idxmax()+1):
#        f.readline()
#    with open(outfilename, "a") as o:
#        o.write(f.readline())

