# MSCRNN
This repository contains data and code for the paper [One Size Does Not Fit All: Multi-Scale, Cascaded RNNs for Radar Classification](https://dl.acm.org/doi/abs/10.1145/3360322.3360860).

1. Kindly download the data here: https://doi.org/10.5281/zenodo.3451408

2. Fix the data path: https://github.com/dhruboroy29/MSCRNN/blob/master/joint_training/joint_EMI_winlen_256_slurm.py#L15

3. Create training scripts by running https://github.com/dhruboroy29/MSCRNN/blob/master/joint_training/joint_EMI_winlen_256_slurm.py

4. Submit SLURM training jobs: https://github.com/dhruboroy29/MSCRNN/blob/master/joint_training/submit.sh

5. Collect postprocessing results: https://github.com/dhruboroy29/MSCRNN/blob/master/joint_training/postprocess_all.sh

Kindly cite this work as:

```
@inproceedings{roy2019one,
  title={One size does not fit all: Multi-scale, cascaded RNNs for radar classification},
  author={Roy, Dhrubojyoti and Srivastava, Sangeeta and Kusupati, Aditya and Jain, Pranshu and Varma, Manik and Arora, Anish},
  booktitle={Proceedings of the 6th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation},
  pages={1--10},
  year={2019}
}
```
The code for this project is based on [Microsoft Research India's EdgeML repository](https://github.com/Microsoft/EdgeML).



