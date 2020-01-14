from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import itertools
import ast
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('-type', type=str, default='tar', help='tar/act/any other prefix')
parser.add_argument('-base', type=str, help='Base path of data')
parser.add_argument('-bat', type=str, default='slurm', help='Batch system (pbs/slurm)')
parser.add_argument('-O', type=int, default=3, help='Number of outputs')
parser.add_argument('-ots', type=int, help='Original number of time steps')
parser.add_argument('-rnd', type=int, default=5, help='Number of rounds')
parser.add_argument('-ep', type=int, default=50, help='Number of epochs')
parser.add_argument('-it', type=int, default=10, help='Number of iterations')
parser.add_argument('-k', type=int, default=10,
                    help='Min. number of consecutive target instances. 100 for max possible')
parser.add_argument('-q15', type=ast.literal_eval, default=False, help='Is this a Q15 gridsearch?')

if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()

if args.q15:
    p0 = ['quantSigm', 'quantTanh']  # ['relu','quantSigm','quantTanh'] #['quantSigm']
    p1 = ['quantTanh', 'quantSigm']  # ['relu','quantSigm','quantTanh'] #['quantTanh']
else:
    p0 = ['sigmoid', 'tanh']
    p1 = ['tanh', 'sigmoid']

p2 = [64, 128]
p3 = [16, 32, 64]

out_folder = os.path.join('..', args.bat + '_hpc')
out_suffix = ''
if args.q15:
    out_suffix += '_q15'
out_file = os.path.join('..', args.bat + '_hpc', args.type + out_suffix + '.sh')


def generate_trainstring(v):
    res_str = "python3 ../tf/examples/EMI-RNN/step2_2tier_joint_tvt.py -O " + str(args.O) \
              + " -gN " + str(v[0]) + " -uN " + str(v[1]) + " -bs " + str(v[2]) + " -H " + str(
        v[3]) + " -Dat " + args.base \
              + " -rnd " + str(args.rnd) + " -it " + str(args.it) + " -ep " + str(args.ep) + " -ots " + str(args.ots) + " -k " + str(args.k) + " -out $outname"

    return res_str


pool = ThreadPool()
hyperparams = np.asarray(list(itertools.product(p0, p1, p2, p3))).tolist()
results = pool.map(generate_trainstring, hyperparams)

# Flatten
# results = [item for sublist in results for item in sublist]

with open(out_file, 'w') as f:
    print('outname=`echo $0 | sed "s/.sh/.out/g"`', file=f)
with open(out_file, 'a') as f:
    print(*results, sep="\n", file=f)
