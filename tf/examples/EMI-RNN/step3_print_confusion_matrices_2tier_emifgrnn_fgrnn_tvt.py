import os
import errno
import sys
import tensorflow as tf
import numpy as np
import argparse
import time
import csv

# Making sure edgeml is part of python path
sys.path.insert(0, '../tf/')
sys.path.insert(0, 'tf/')
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

np.random.seed(42)
tf.set_random_seed(42)

# FastGRNN and FastRNN imports
import edgeml.utils as utils
from edgeml.graph.rnn import EMI_DataPipeline
from edgeml.graph.rnn import EMI_FastGRNN
from edgeml.graph.rnn import FastGRNNCell
from edgeml.trainer.emirnnTrainer import EMI_Trainer_2Tier, EMI_Driver
import edgeml.utils

parser = argparse.ArgumentParser(description='HyperParameters for EMI-FastGRNN')
parser.add_argument('-k', type=int, default=2, help='Min. number of consecutive target instances. 100 for max possible')
parser.add_argument('-H', type=int, default=16, help='Number of hidden units')
parser.add_argument('-H2', type=int, default=32, help='Number of second-tier hidden units')
parser.add_argument('-ts', type=int, default=48, help='Number of timesteps')
parser.add_argument('-ots', type=int, default=256, help='Original number of timesteps')
parser.add_argument('-F', type=int, default=2, help='Number of features')
parser.add_argument('-fb', type=float, default=1.0, help='Forget bias')
parser.add_argument('-O', type=int, default=3, help='Number of outputs - all (#target types+noise)')
parser.add_argument('-d', type=bool, default=False, help='Dropout?')
parser.add_argument('-kp', type=float, default=0.9, help='Keep probability')
parser.add_argument('-uN', type=str, default="quantTanh", help='Update nonlinearity')
parser.add_argument('-gN', type=str, default="quantSigm", help='Gate nonlinearity')
parser.add_argument('-wR', type=int, default=5, help='Rank of W')
parser.add_argument('-uR', type=int, default=6, help='Rank of U')
parser.add_argument('-bs', type=int, default=32, help='Batch size')
parser.add_argument('-ep', type=int, default=3, help='Number of epochs per iteration')
parser.add_argument('-it', type=int, default=4, help='Number of iterations per round')
parser.add_argument('-rnd', type=int, default=10, help='Number of rounds')
parser.add_argument('-Dat', type=str, help='Data directory')

args = parser.parse_args()

# Network parameters for our FastGRNN + FC Layer
k = args.k #2
NUM_HIDDEN = args.H #16
NUM_HIDDEN_SECONDTIER = args.H2 #16
NUM_TIMESTEPS = args.ts #48
ORIGINAL_NUM_TIMESTEPS = args.ots #256
NUM_FEATS = args.F #2
FORGET_BIAS = args.fb #1.0
NUM_OUTPUT = args.O #2
USE_DROPOUT = args.d #False
KEEP_PROB = args.kp #0.9

# Non-linearities can be chosen among "tanh, sigmoid, relu, quantTanh, quantSigm"
UPDATE_NL = args.uN #"quantTanh"
GATE_NL = args.gN #"quantSigm"

# Ranks of Parameter matrices for low-rank parameterisation to compress models.
WRANK = args.wR #5
URANK = args.uR #6

# For dataset API
PREFETCH_NUM = 5
BATCH_SIZE = args.bs #32

# Number of epochs in *one iteration*
NUM_EPOCHS = args.ep #3
# Number of iterations in *one round*. After each iteration,
# the model is dumped to disk. At the end of the current
# round, the best model among all the dumped models in the
# current round is picked up..
NUM_ITER = args.it #4
# A round consists of multiple training iterations and a belief
# update step using the best model from all of these iterations
NUM_ROUNDS = args.rnd #10
#LEARNING_RATE=0.001

# A staging directory to store models
MODEL_PREFIX = '/tmp/models/model-fgrnn/'+str(int(time.time()))+'/'

# Make model directory
try:
    os.makedirs(MODEL_PREFIX)
except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(MODEL_PREFIX):
        pass
    else:
        raise

# Loading the data
data_dir = args.Dat #'/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Displacement_Detection/Data/Austere_subset_features/' \
           #'Raw_winlen_256_stride_171/48_16/'

x_train, y_train = np.load(os.path.join(data_dir,'x_train.npy')), np.load(os.path.join(data_dir,'y_train.npy'))
x_test, y_test = np.load(os.path.join(data_dir,'x_test.npy')), np.load(os.path.join(data_dir,'y_test.npy'))
x_val, y_val = np.load(os.path.join(data_dir,'x_val.npy')), np.load(os.path.join(data_dir,'y_val.npy'))

# BAG_TEST, BAG_TRAIN, BAG_VAL represent bag_level labels. These are used for the label update
# step of EMI/MI RNN
BAG_TEST = np.argmax(y_test[:, 0, :], axis=1)
SUBINST_TEST = np.argmax(y_test, axis=2)
BAG_TRAIN = np.argmax(y_train[:, 0, :], axis=1)
SUBINST_TRAIN = np.argmax(y_train, axis=2)
BAG_VAL = np.argmax(y_val[:, 0, :], axis=1)
SUBINST_VAL = np.argmax(y_val, axis=2)
NUM_SUBINSTANCE = x_train.shape[1]
print("x_train shape is:", x_train.shape)
print("y_train shape is:", y_train.shape)
print("x_test shape is:", x_val.shape)
print("y_test shape is:", y_val.shape)

# Adjustment for max k: number of subinstances
if k==100:
    k = x_train.shape[1]

# Create upper FastGRNN cell
upperFastGRNN = FastGRNNCell(NUM_HIDDEN_SECONDTIER, wRank=WRANK, uRank=URANK,
                           gate_non_linearity=GATE_NL, update_non_linearity=UPDATE_NL)


# Define the upper layer - This incorporates upper FastGRNN
# Define the two classifiers at two levels
def createExtendedGraph(self, baseOutput, *args, **kwargs):
    # Get EMI output - Target vs noise, so NUM_OUTPUT hardcoded as 2
    W1 = tf.Variable(np.random.normal(size=[NUM_HIDDEN, 2]).astype('float32'), name='W1')
    B1 = tf.Variable(np.random.normal(size=[2]).astype('float32'), name='B1')
    y_cap = tf.add(tf.tensordot(baseOutput, W1, axes=1), B1, name='y_cap_tata')

    # Get EMI embeddings
    emiEmbeddings = baseOutput[:, :, -1, :]

    # Unroll second tier, get final hidden state
    x = tf.unstack(emiEmbeddings, np.shape(emiEmbeddings)[1], 1)
    outputs, states = tf.nn.static_rnn(upperFastGRNN, x, dtype=tf.float32)
    secondtier=outputs[-1]

    # Get second-tier output - here, #outputs is 3, only noise loss doesn't propagate up
    W2 = tf.Variable(np.random.normal(size=[NUM_HIDDEN_SECONDTIER, NUM_OUTPUT]).astype('float32'), name='W2')
    B2 = tf.Variable(np.random.normal(size=[NUM_OUTPUT]).astype('float32'), name='B2')
    y_cap_upper = tf.add(tf.tensordot(secondtier, W2, axes=1), B2, name='y_cap_upper')
    self.output = [y_cap, y_cap_upper]
    self.graphCreated = True


def restoreExtendedGraph(self, graph, *args, **kwargs):
    y_cap = graph.get_tensor_by_name('y_cap_tata:0')
    y_cap_upper = graph.get_tensor_by_name('y_cap_upper:0')
    self.output = [y_cap, y_cap_upper]

    self.graphCreated = True


def feedDictFunc(self, keep_prob=None, inference=False, **kwargs):
    if inference is False:
        feedDict = {self._emiGraph.keep_prob: keep_prob}
    else:
        feedDict = {self._emiGraph.keep_prob: 1.0}
    return feedDict


EMI_FastGRNN._createExtendedGraph = createExtendedGraph
EMI_FastGRNN._restoreExtendedGraph = restoreExtendedGraph
if USE_DROPOUT is True:
    EMI_FastGRNN.feedDictFunc = feedDictFunc

inputPipeline = EMI_DataPipeline(NUM_SUBINSTANCE, NUM_TIMESTEPS, NUM_FEATS, NUM_OUTPUT)
emiFastGRNN = EMI_FastGRNN(NUM_SUBINSTANCE, NUM_HIDDEN, NUM_TIMESTEPS, NUM_FEATS, wRank=WRANK, uRank=URANK,
                           gate_non_linearity=GATE_NL, update_non_linearity=UPDATE_NL, useDropout=USE_DROPOUT)
emiTrainer2tier = EMI_Trainer_2Tier(NUM_TIMESTEPS, numOutputUpper=NUM_OUTPUT, lossType='xentropy')


# Connect elementary parts together to create forward graph
tf.reset_default_graph()
g1 = tf.Graph()
with g1.as_default():
    # Obtain the iterators to each batch of the data
    x_batch, y_batch = inputPipeline()

    '''Change instance outputs to 1-hot 2-class: Noise vs Target'''
    # Deconvert instances from one-hot
    y_batch_1hotdecoded = tf.argmax(y_batch, axis=2)
    # Change all non zeros to target class 1 for EMI
    mask = tf.greater(y_batch_1hotdecoded, tf.zeros_like(y_batch_1hotdecoded))
    y_batch_lower = tf.cast(mask, tf.int32)
    # Convert back to 2-class one-hot
    y_batch_lower = tf.one_hot(y_batch_lower, depth=2, axis=-1)

    '''Change bag outputs to 1-hot 3-class: Noise vs Human vs Nonhuman'''
    # Get bag outputs from batch
    y_batch_upper = tf.argmax(y_batch[:, 0, :], axis=1)
    # Convert to one-hot
    y_batch_upper = tf.one_hot(y_batch_upper, depth=NUM_OUTPUT, axis=-1)

    # Create the forward computation graph based on the iterators
    y_cap, y_cap_upper = emiFastGRNN(x_batch)
    # Create loss graphs and training routines
    emiTrainer2tier(y_cap, y_cap_upper, y_batch_lower, y_batch_upper)


with g1.as_default():
    emiDriver = EMI_Driver(inputPipeline, emiFastGRNN, emiTrainer2tier)

emiDriver.initializeSession(g1, config=config)
#print('NOTE: TRAINING TIME ACCURACIES SHOWN ARE EMI ONLY')
#y_updated, modelStats = emiDriver.run(numClasses=2, x_train=x_train,
#                                      y_train=y_train, bag_train=BAG_TRAIN,
#                                      x_val=x_val, y_val=y_val, bag_val=BAG_VAL,
#                                      numIter=NUM_ITER, keep_prob=KEEP_PROB,
#                                      numRounds=NUM_ROUNDS, batchSize=BATCH_SIZE,
#                                      numEpochs=NUM_EPOCHS, modelPrefix=MODEL_PREFIX,
#                                      fracEMI=0.5, updatePolicy='top-k', k=k)

'''
Evaluating the  trained model
'''

# Early Prediction Policy: We make an early prediction based on the predicted classes
#     probability. If the predicted class probability > minProb at some step, we make
#     a prediction at that step.
def earlyPolicy_minProb(instanceOut, minProb, **kwargs):
    assert instanceOut.ndim == 2
    classes = np.argmax(instanceOut, axis=1)
    prob = np.max(instanceOut, axis=1)
    index = np.where(prob >= minProb)[0]
    if len(index) == 0:
        assert (len(instanceOut) - 1) == (len(classes) - 1)
        return classes[-1], len(instanceOut) - 1
    index = index[0]
    return classes[index], index

def getEarlySaving(predictionStep, numTimeSteps, returnTotal=False):
    predictionStep = predictionStep + 1
    predictionStep = np.reshape(predictionStep, -1)
    totalSteps = np.sum(predictionStep)
    maxSteps = len(predictionStep) * numTimeSteps
    savings = 1.0 - (totalSteps / maxSteps)
    if returnTotal:
        return savings, totalSteps
    return savings

#k = 2
#predictions, predictionStep = emiDriver.getInstancePredictions(x_test, y_test, earlyPolicy_minProb,
#                                                               minProb=0.99, keep_prob=1.0)
#bagPredictions = emiDriver.getBagPredictions(predictions, minSubsequenceLen=k, numClass=NUM_OUTPUT)
#print('Accuracy at k = %d: %f' % (k,  np.mean((bagPredictions == BAG_TEST).astype(int))))
#mi_savings = (1 - NUM_TIMESTEPS / ORIGINAL_NUM_TIMESTEPS)
#emi_savings = getEarlySaving(predictionStep, NUM_TIMESTEPS)
#total_savings = mi_savings + (1 - mi_savings) * emi_savings
#print('Savings due to MI-RNN : %f' % mi_savings)
#print('Savings due to Early prediction: %f' % emi_savings)
#print('Total Savings: %f' % (total_savings))


# A slightly more detailed analysis method is provided.
#df = emiDriver.analyseModel(predictions, BAG_TEST, NUM_SUBINSTANCE, NUM_OUTPUT)

# Write model stats file
#with open(os.path.join(data_dir,'modelstats_H=' + str(NUM_HIDDEN) + '_k=' + str(k) + '_ep='+ str(NUM_EPOCHS)
#                                + '_it=' + str(NUM_ITER) + '_rnd=' + str(NUM_ROUNDS) + '.csv'),'w') as out:
#    csv_out=csv.writer(out)
#    csv_out.writerow(['name','num'])
#    for row in modelStats:
#        csv_out.writerow(row)

# Pick the best model
devnull = open(os.devnull, 'r')
acc = 0.0

with open(os.path.join(data_dir,'modelstats_2tier_H=' + str(NUM_HIDDEN) + "_H2=" + str(NUM_HIDDEN_SECONDTIER)
                                + '_k=' + str(k) + '_ep='+ str(NUM_EPOCHS)
                                + '_it=' + str(NUM_ITER) + '_rnd=' + str(NUM_ROUNDS)
                                + '_bs=' + str(BATCH_SIZE) + '.csv'),'r') as stats_csv:
    modelStats = csv.reader(stats_csv)
    header = next(modelStats)
    for val in modelStats:
        c_round_, c_acc, c_modelPrefix, c_globalStep = val
        c_round_ = int(c_round_)
        c_globalStep = int(c_globalStep)

        '''Run each model on validation set'''
        # Get instance-level predictions
        emiDriver.loadSavedGraphToNewSession(c_modelPrefix, int(c_globalStep), redirFile=devnull)
        predictions, predictionStep = emiDriver.getInstancePredictions(x_val, y_val, earlyPolicy_minProb,
                                                                       minProb=0.99, keep_prob=1.0)
        # Get bag-level predictions
        bagPredictions = emiDriver.getBagPredictions(predictions, k=k, numClass=2)
        # Get upper tier predictions
        upperPredictions = emiDriver.getUpperTierPredictions(x_val, y_val)
        # Get validation predictions following switch emulation: consider top level prediction only when bottom level output nonzero
        valPredictions = np.multiply(bagPredictions, upperPredictions)
        # Finally, compute validation accuracy
        val_acc = np.mean((valPredictions == BAG_VAL).astype(int))

        print("VALIDATION CONF MATRIX\tRound: %2d, window length: %3d, Validation accuracy: %.4f" % (c_round_, ORIGINAL_NUM_TIMESTEPS, val_acc), end='')

        # Print confusion matrix
        print('\n')
        valcmatrix = utils.getConfusionMatrix(valPredictions, BAG_VAL, NUM_OUTPUT)
        utils.printFormattedConfusionMatrix(valcmatrix)
        print('\n')

        '''Run the model on the test set'''
        # Get instance-level predictions
        predictions, predictionStep = emiDriver.getInstancePredictions(x_test, y_test, earlyPolicy_minProb,
                                                                       minProb=0.99, keep_prob=1.0)
        # Get bag-level predictions
        bagPredictions = emiDriver.getBagPredictions(predictions, k=k, numClass=2)
        # Get upper tier predictions
        upperPredictions = emiDriver.getUpperTierPredictions(x_test, y_test)
        # Get validation predictions following switch emulation: consider top level prediction only when bottom level output nonzero
        testPredictions = np.multiply(bagPredictions, upperPredictions)

        print("TEST CONF MATRIX\tRound: %2d, window length: %3d, Validation accuracy: %.4f" % (c_round_, ORIGINAL_NUM_TIMESTEPS, acc),
              end='')
        print(', Test Accuracy (k = %d): %f, ' % (k, np.mean((testPredictions == BAG_TEST).astype(int))), end='')

        # Print confusion matrix
        print('\n')
        testcmatrix = utils.getConfusionMatrix(testPredictions, BAG_TEST, NUM_OUTPUT)
        utils.printFormattedConfusionMatrix(testcmatrix)
        print('\n')

        if val_acc > acc:
            round_, acc, modelPrefix, globalStep = c_round_, val_acc, c_modelPrefix, c_globalStep

''' Compute test accuracy on best model obtained above'''
print("*******\t BEST MODEL TEST CONFUSION MATRIX\t*******")
# Get instance-level predictions
emiDriver.loadSavedGraphToNewSession(modelPrefix, int(globalStep), redirFile=devnull)
predictions, predictionStep = emiDriver.getInstancePredictions(x_test, y_test, earlyPolicy_minProb,
                                                           minProb=0.99, keep_prob=1.0)
# Get bag-level predictions
bagPredictions = emiDriver.getBagPredictions(predictions, k=k, numClass=2)
# Get upper tier predictions
upperPredictions = emiDriver.getUpperTierPredictions(x_test, y_test)
# Get validation predictions following switch emulation: consider top level prediction only when bottom level output nonzero
testPredictions = np.multiply(bagPredictions, upperPredictions)

print("Round: %2d, window length: %3d, Validation accuracy: %.4f" % (round_, ORIGINAL_NUM_TIMESTEPS, acc), end='')
print(', Test Accuracy (k = %d): %f, ' % (k,  np.mean((testPredictions == BAG_TEST).astype(int))), end='')

# Print confusion matrix
print('\n')
testcmatrix = utils.getConfusionMatrix(testPredictions, BAG_TEST, NUM_OUTPUT)
utils.printFormattedConfusionMatrix(testcmatrix)
print('\n')

# Print model size
metaname = modelPrefix + '-%d.meta' % globalStep
utils.getModelSize(metaname)

mi_savings = (1 - NUM_TIMESTEPS / ORIGINAL_NUM_TIMESTEPS)
emi_savings = getEarlySaving(predictionStep, NUM_TIMESTEPS)
total_savings = mi_savings + (1 - mi_savings) * emi_savings
print('Savings due to MI-RNN : %f' % mi_savings)
print('Additional savings due to Early prediction: %f' % emi_savings)
print("Total Savings: %f" % total_savings)
