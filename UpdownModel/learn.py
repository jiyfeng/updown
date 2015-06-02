## learn.py
## Author: Yangfeng Ji
## Date: 10-01-2014
## Time-stamp: <yangfeng 11/20/2014 13:49:18>

""" Mini-batch SGD learning on up-down semantic composition
    model
"""

from addpath import checkpath
checkpath()

import gzip, sys
from cPickle import load, dump
from random import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix
from time import localtime, strftime
from SemComp.tree import *
from SemComp.datastructure import *
from SemComp.vocab import Vocab

def fscore(mat):
    p = mat[1,1] / (mat[0,1] + mat[1,1] + 1e-7)
    r = mat[1,1] / (mat[1,0] + mat[1,1] + 1e-7)
    return (2*p*r)/(p+r+1e-7)
    

class SGDLearn(object):
    def __init__(self, model, vocab, T=1000, lr=0.05, reg=0.0,
                 normthresh=5.0):
        """
        """
        self.model, self.vocab = model, vocab
        self.T, self.lr, self.reg = T, lr, reg
        self.normthresh = normthresh
        self.bestcesum, self.bestacc = 100.0, 0.0
        self.bestfscore = 0.0
        self.truelabels = None
        self.predlabels = None


    def createsample(self, sample):
        """ Create sample
            1, build upward tree pair
            2, find matching entity node pair
            3, tree transformation for entity nodes
        """
        ridx = sample['ridx']
        Lt, Rt = Tree(), Tree()
        # print '----------------------------------------'
        # print sample['left_tree']
        # print sample['right_tree']
        try:
            Lt.build(sample['left_align'], self.vocab)
            Lt.find_entitynodes()
            Rt.build(sample['right_align'], self.vocab)
            Rt.find_entitynodes()
            # print Lt.entity_nodedict
            # print Rt.entity_nodedict
            sample = Sample(ridx, (Lt, Rt))
        except BuildTreeError:
            sample = None
        return sample


    def minibatch_learn(self, trnSample, minibatchsize=32,
            evalstep=1, devSample=None, updateparam=True,
            updateup=True, updatedown=True,
            fmodel="tmp-model.pickle.gz"):
        """ Learning with mini-batch training examples

        :type trnSample:
        :param trnSample:

        :type minibatchsize: int
        :param minibatchsize: mini-batch size
        """
        samplelist = []
        counter, N = 0, len(trnSample)
        self.model.updateparam = True
        self.model.updateup = True
        self.model.updatedown = True
        for niter in range(self.T):
            while len(samplelist) < minibatchsize:
                sample = self.createsample(trnSample[counter])
                counter += 1
                if counter >= N:
                    counter = counter % N
                    shuffle(trnSample)
                if sample is not None:
                    samplelist.append(sample)
            # Gradient
            grad = self.model.grad_minibatch(samplelist)
            samplelist = [] # Empty list
            # Update all parameters
            self.model.update_adagrad(grad, self.lr, self.reg,
                normthresh=self.normthresh)
            # Print out information
            # print self.model.ParamAll['Param']['down'].L.sum()
            if niter % evalstep == 0:
                timestamp = strftime("%a, %d %b %Y %H:%M:%S", localtime())
                print '{}: {} iterations'.format(timestamp, niter)
                if devSample is not None:
                    acc, fscore, cesum = self.test(devSample)
                    # print acc, cesum
                    print '\tObjective = {0:.4f} ({1:.4f}), Accuracy = {2:.4f} ({3:.4f}), Fscore = {4:.4f} ({5:.4f})'.format(cesum,
                        self.bestcesum, acc, self.bestacc, fscore, self.bestfscore)
                if cesum < self.bestcesum:
                    self.bestcesum = cesum
                if fscore > self.bestfscore:
                    self.bestfscore = fscore
                if acc > self.bestacc:
                    self.bestacc = acc
                    self.model.savemodel(fmodel)
                    fresult = fmodel.replace("pickle.gz","result.pickle.gz")
                    self.saveresult(fresult)


    def saveresult(self, fname):
        """
        """
        print 'Save results into file: {}'.format(fname)
        with gzip.open(fname, 'w') as fout:
            data = {'true':self.truelabels, 'pred':self.predlabels}
            dump(data, fout)


    def test(self, tstSample):
        """ Test performance on tstSample
        """
        tstcounter, cesum = 0.0, 0.0
        predlabels = numpy.zeros((len(tstSample),))
        truelabels = numpy.zeros((len(tstSample),))
        for (sidx, sample) in enumerate(tstSample):
            sample = self.createsample(sample)
            if sample is None:
                continue
            tstcounter += 1.0
            prob = self.model.predict(sample)[0]
            cesum += self.model.objective(prob, sample.ridx)
            # if tstcounter == 1.0:
                # print 'ridx = {}'.format(sample.ridx)
                # print 'prob = {}'.format(prob)
            predidx = numpy.argmax(prob)
            predlabels[sidx] = predidx
            truelabels[sidx] = sample.ridx
        self.truelabels, self.predlabels = truelabels, predlabels
        acc = accuracy_score(truelabels, predlabels)
        mat = confusion_matrix(truelabels, predlabels)
        f1 = fscore(mat)
        # print mat
        return acc, f1, (cesum / tstcounter)
            
