# Script to get windowed radar data, generated by:
# https://github.com/dhruboroy29/MATLAB_Scripts/blob/neel/Scripts/extract_target_windows.m

import os
import numpy as np
from helpermethods import ReadRadarWindows, one_hot, bagData
import argparse

np.random.seed(42)

def getRadarData(path):
    # Shuffle
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    noise_data = ReadRadarWindows(os.path.join(path, 'Noise'))
    noise_label = np.array([0] * len(noise_data))

    targets_data = np.concatenate((ReadRadarWindows(os.path.join(path, 'Human')),
                                  ReadRadarWindows(os.path.join(path, 'Nonhuman'))))
    targets_label = np.array([1] * len(targets_data))
    X = np.concatenate([targets_data, noise_data])
    y = np.concatenate([targets_label, noise_label])

    # Shuffle
    return unison_shuffled_copies(X,y)

parser = argparse.ArgumentParser(description='HyperParameters for EMI-LSTM')
#parser.add_argument('-Dat', type=str, help='Directory containing fixed-length windowed data')
parser.add_argument('-l', type=int, default=48, help='Sub-instance length')
parser.add_argument('-s', type=int, default=16, help='Sub-instance stride length')
parser.add_argument('-spl', type=float, default=0.2, help='Validation/test split')

args = parser.parse_args()

base_dir = '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/FastGRNN/Data/Bumblebee/'
list_dirs = ['bb_3class_winlen_256_winindex_all', 'bb_3class_winlen_384_winindex_all', 'bb_3class_winlen_512_winindex_all',
             'bb_3class_winlen_640_winindex_all', 'bb_3class_winlen_768_winindex_all']

for dir in list_dirs:
    extractedDir = os.path.join(base_dir, dir) #'/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Displacement_Detection/' \
                    #'Data/Austere_subset_features/Raw_winlen_256_stride_171'

    # Train data
    x_train, y_train = getRadarData(os.path.join(extractedDir,dir+'_train'))

    # Validation data
    x_val, y_val = getRadarData(os.path.join(extractedDir, dir + '_val'))

    # Test data
    x_test, y_test = getRadarData(os.path.join(extractedDir, dir + '_test'))

    feats = x_train.shape[-1]
    timesteps = x_train.shape[-2]

    # Splitting data into train/test/validation (size of test set = validation set)
    #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=args.spl, random_state=42)
    #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
    #                                                  test_size=args.spl*(len(y_train)+len(y_test))/len(y_train), random_state=42)

    # Normalize train, test, validation
    x_train = np.reshape(x_train, [-1, feats])
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    # normalize train
    x_train = x_train - mean
    x_train = x_train / std
    x_train = np.reshape(x_train, [-1, timesteps, feats])

    # normalize val
    x_val = np.reshape(x_val, [-1, feats])
    x_val = x_val - mean
    x_val = x_val / std
    x_val = np.reshape(x_val, [-1, timesteps, feats])

    # normalize test
    x_test = np.reshape(x_test, [-1, feats])
    x_test = x_test - mean
    x_test = x_test / std
    x_test = np.reshape(x_test, [-1, timesteps, feats])

    # one-hot encoding of labels
    numOutput = 2
    y_train = one_hot(y_train, numOutput)
    y_val = one_hot(y_val, numOutput)
    y_test = one_hot(y_test, numOutput)


    # Create EMI data
    subinstanceLen = args.l #48
    subinstanceStride = args.s #16
    outDir = extractedDir + '/%d_%d/' % (subinstanceLen, subinstanceStride)

    print('subinstanceLen', subinstanceLen)
    print('subinstanceStride', subinstanceStride)
    print('outDir', outDir)
    try:
        os.mkdir(outDir)
    except OSError:
        exit("Could not create %s" % outDir)
    assert len(os.listdir(outDir)) == 0

    x_bag_train, y_bag_train = bagData(x_train, y_train, subinstanceLen, subinstanceStride,
                                       numClass=numOutput, numSteps=timesteps, numFeats=feats)
    np.save(outDir + '/x_train.npy', x_bag_train)
    np.save(outDir + '/y_train.npy', y_bag_train)
    print('Num train %d' % len(x_bag_train))
    x_bag_test, y_bag_test = bagData(x_test, y_test, subinstanceLen, subinstanceStride,
                                       numClass=numOutput, numSteps=timesteps, numFeats=feats)
    np.save(outDir + '/x_test.npy', x_bag_test)
    np.save(outDir + '/y_test.npy', y_bag_test)
    print('Num test %d' % len(x_bag_test))
    x_bag_val, y_bag_val = bagData(x_val, y_val, subinstanceLen, subinstanceStride,
                                       numClass=numOutput, numSteps=timesteps, numFeats=feats)
    np.save(outDir + '/x_val.npy', x_bag_val)
    np.save(outDir + '/y_val.npy', y_bag_val)
    print('Num val %d' % len(x_bag_val))


print('\nAll done!')