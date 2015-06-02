## model.py
## Author: Yangfeng Ji
## Date: 08-21-2014
## Time-stamp: <yangfeng 09/24/2014 21:46:35>

"""
Build the classification model for discourse
relation identification.
Refer to the note for more technical details
--- YJ
"""
from addpath import checkpath
checkpath()

import numpy, gzip
from numpy.linalg import norm
from SemComp.tree import *
from util import *
from cPickle import load, dump
from upwardmodel import UpwardModel

rng = numpy.random.RandomState(1234)

class UpwardModelDO(UpwardModel):
    def __init__(self, nDim, nRela):
        """ Initialize the basic model, which is the standard
            RNN based semantic composition model

        :type nDim: int
        :param nDim: latent units

        :type nRela: int
        :param nRela: number of discourse relations
        """
        UpwardModel.__init__(self, nDim, nRela)
        self.Umask = {}
        for nr in range(nRela):
            self.Umask[nr] = rng.binomial(1, 0.5, size=self.U[nr].shape)


    def combine(self, sample, dropout):
        """ Combine left/right latent vec to get a un-normalized
            prob
        """
        dpU = self.dropout()
        treepair = sample.treepair
        left_vec = treepair[0].upward(self.Param)
        right_vec = treepair[1].upward(self.Param)
        # Compute unnormalized prob
        prob = numpy.zeros((len(dpU),))
        for (idx, Ui) in dpU.iteritems():
            prob[idx] = left_vec.T.dot(Ui.dot(right_vec)) + self.b[idx]
        return prob, left_vec, right_vec


    def dropout(self):
        """ Randomly set some weights to be 0
        """
        dropoutU = {}
        for (idx, U_idx) in self.U.iteritems():
            U_idx *= self.Umask[idx]
            dropoutU[idx] = U_idx
        return dropoutU
    

    def grad(self, sample):
        """ Gradient of parameters wrt the given sample

        Get the gradient of vocab also from this function,
        but doesn't implement the update in this class

        :type sample: instance of Sample class
        :param sample: see datastructure for more info
        """
        prob, vl, vr = self.predict(sample) # Output prob
        tgt = numpy.zeros(prob.shape)
        tgt[sample.ridx] = 1.0
        gradCE_x = prob - tgt # (nRela,)
        gradall = self.grad_given_ce(sample, gradCE_x, vl, vr)
        for (idx, U) in gradall['U'].iteritems():
            gradall['U'][idx] = U * self.Umask[idx]
        return gradall

    
