## learn.py
## Author: Yangfeng Ji
## Date: 08-27-2014
## Time-stamp: <yangfeng 11/19/2014 20:17:21>

""" Mini-batch SGD learning on upward semantic composition
    model
"""

from addpath import checkpath
checkpath()

import gzip
from time import localtime, strftime
from SemComp.tree import *
from SemComp.datastructure import *
from SemComp.vocab import Vocab
from util import *
from random import shuffle
from sklearn.metrics import accuracy_score
from cPickle import dump

class SGDLearn:
    def __init__(self, model, vocab, T=1000, lr=0.1, reg=0.0,
                 norm_thresh=1.0, updatevocab=False):
        """ Initialization

        :type lr: float
        :param lr: learning rate

        :type reg: float
        :param reg: regularization parameter
        """
        self.model = model
        self.lr = lr
        self.reg = reg
        self.T = T
        self.norm_thresh = norm_thresh
        self.bestperf, self.bestce_sum = 0.0, 100.0
        self.vocab = vocab
        self.updatevocab = updatevocab


    def createsample(self, sample):
        Lt, Rt = Tree(), Tree()
        try:
            Lt.build(sample['left_tree'], self.vocab)
            Rt.build(sample['right_tree'], self.vocab)
            sample = Sample(sample['ridx'], (Lt, Rt))
            return sample
        except BuildTreeError:
            return None
    

    def minibatch_learn(self, trnSample, minibatch_size=32,
        evalstep=1,devSample=None,fmodel='tmp-model.pickle.gz'):
        """ Training model with one pass on all training samples

        :type trnSample: list of dict
        :param trnSample: list of training sample

        :type vocab: instance of Vocab
        :param vocab: word vocab and vector representation

        :type epoch_idx: int
        :param epoch_idx: epoch index
        """
        sample_list = []
        counter, N = 0, len(trnSample)
        for niter in range(self.T):
            while (len(sample_list) < minibatch_size):
                # Create mini-batch sample list
                sample = self.createsample(trnSample[counter])
                # Reset the counter and shuffle the examples
                counter += 1
                if counter >= N:
                    counter = counter % N
                    shuffle(trnSample)
                if sample is not None:
                    sample_list.append(sample)
            # Compute gradient
            grad = self.model.grad_minibatch(sample_list)
            sample_list = [] # Empty the list
            # Update parameters
            self.model.update_adagrad(grad, self.lr, self.reg,
                updateparam=True, updatecomp=True,
                norm_thresh=self.norm_thresh)
            # Update word representation
            if self.updatevocab:
                print '\tUpdate vocab ...'
                self.vocab.update_adagrad(grad['Word'],
                    self.lr, self.reg,
                    norm_thresh=self.norm_thresh)
            # Print out information
            if niter % evalstep == 0:
                time_stamp = strftime("%a, %d %b %Y %H:%M:%S", localtime())
                print '{}: {} iterations'.format(time_stamp, niter)
                if devSample is not None:
                    acc, ce_sum, tlabels, plabels = self.test(devSample)
                    print '\tObjective = {0:.4f} ({3:.4f}), Accuracy = {1:.4f} ({2:.4f})'.format(ce_sum,
                        acc, self.bestperf, self.bestce_sum)
                if ce_sum < self.bestce_sum:
                    self.bestce_sum = ce_sum
                if acc > self.bestperf:
                    self.bestperf = acc
                    print 'Save model and the best performance'
                    self.model.savemodel(fmodel)
                    fresult = fmodel.replace('pickle.gz', 'result.pickle.gz')
                    self.saveresult(fresult, tlabels, plabels)


    def saveresult(self, fname, tlabels, plabels):
        """
        """
        print 'Save results into file: {}'.format(fname)
        with gzip.open(fname, 'w') as fout:
            data = {'true':tlabels, 'pred':plabels}
            dump(data, fout)


    def test(self, tstSample):
        """ Test 
        """
        tstcounter, ce_sum = 0.0, 0.0
        predlabels = numpy.zeros((len(tstSample),))
        truelabels = numpy.zeros((len(tstSample),))
        for (sidx, sample) in enumerate(tstSample):
            sample = self.createsample(sample)
            if sample is None:
                continue
            tstcounter += 1
            prob = self.model.predict(sample, dropout=False)[0]
            ce = self.model.cross_entropy(prob, sample.ridx)
            pred_ridx = numpy.argmax(prob)
            ce_sum += ce
            predlabels[sidx] = pred_ridx
            truelabels[sidx] = sample.ridx
        acc = accuracy_score(truelabels, predlabels)
        return acc, (ce_sum / tstcounter), truelabels, predlabels
                
