## model.py
## Author: Yangfeng Ji
## Date: 08-21-2014
## Time-stamp: <yangfeng 09/28/2014 14:20:38>

"""
In this module, I change the objective function from
cross-entropy function to max-margin function
--- YJ
"""

from addpath import checkpath
checkpath()

import numpy, gzip
from numpy.linalg import norm
from SemComp.tree import *
from util import *
from cPickle import load, dump
from upwardmodelmv import UpwardModelMV

class UpwardModelMM(UpwardModelMV):
    def __init__(self, nDim, nRela, rng=None):
        """ Initialize the basic model, which is the standard
            RNN based semantic composition model

        :type nDim: int
        :param nDim: latent units

        :type nRela: int
        :param nRela: number of discourse relations

        :type rng: numpy.random.RandomSate
        :param rng: random state for initialization
        """
        if rng is None:
            rng = numpy.random.RandomState(1234)
        UpwardModelMV.__init__(self, nDim, nRela, rng)


    def cross_entropy(self, prob, ridx):
        """ Max-margin objective function for a given sample

            To use the same learning code, let's still call
            it cross_entropy :-)

            Note that, prob <= 0
        """
        obj = 0.0
        for idx in range(prob.shape[0]):
            if idx != ridx:
                obj += max(0, 1 - prob[ridx] + prob[idx])
        return obj


    def grad(self, sample):
        """ Grad wrt to max-margin objective function
        """
        prob, vl, vr = self.predict(sample)
        gradCE_x = numpy.zeros(prob.shape)
        ridx = sample.ridx
        for idx in range(prob.shape[0]):
            if (idx != ridx) and ((prob[ridx]-prob[idx])<1.0):
                gradCE_x[idx] = 1.0
        if gradCE_x.sum() > 0.0:
            gradCE_x /= (prob.shape[0] - 1)
            gradCE_x[ridx] = -gradCE_x.sum()
            # print 'gradCE_x = ', gradCE_x
            gradall = self.grad_given_ce(sample, gradCE_x, vl, vr)
        else:
            # Skip calling RNNs to save some time
            gradall, gradU = {}, {}
            for (c, Uc) in self.U.iteritems():
                gradU[c] = numpy.zeros(Uc.shape)
            gradall['U'] = gradU
            gradall['b'] = numpy.zeros(self.b.shape)
            gradL = numpy.zeros(self.Param.L.shape)
            gradR = numpy.zeros(self.Param.R.shape)
            gradbias = numpy.zeros(self.Param.bias.shape)
            gradparam = Param(gradL, gradR, gradbias)
            gradall['Param'] = gradparam
            gradall['Word'] = {}
        return gradall


    def combine(self, sample, dropout):
        """ Combine left/right latent vec to get a un-normalized
            prob

        :type dropout: bool
        :param dropout: whether use dropout (Not used in here)
        """
        treepair = sample.treepair
        left_vec = treepair[0].upward(self.Param)
        right_vec = treepair[1].upward(self.Param)
        prob = numpy.zeros((len(self.U),))
        for (idx, Ui) in self.U.iteritems():
            vec = numpy.hstack((left_vec, right_vec))
            prob[idx] = Ui.dot(vec) + self.b[idx]
        return prob, left_vec, right_vec
        
        
    def grad_given_ce(self, sample, gradCE_x, vl, vr):
        """ Grad of params given the grad of cross entropy
            The purpose of spliting this part from grad()
            is to make it reusable for joint model

            The basic idea is to get rid of prediction function
        """
        # Grad of CE wrt U_r
        gradU = {}
        for c in range(len(self.U)):
            gradU[c] = gradCE_x[c] * numpy.hstack((vl, vr))
        # Grad of CE wrt b
        gradb = gradCE_x
        # Grad of CE wrt left/right vector
        gradLV = numpy.zeros(vl.shape)
        gradRV = numpy.zeros(vr.shape)
        for c in range(len(self.U)):
            gradLV += gradCE_x[c] * (self.U[c][:self.nDim])
            gradRV += gradCE_x[c] * (self.U[c][self.nDim:(2*self.nDim)])
        # Grad of composition parameters from left/right tree
        # raise ValueError("Double check this part")
        gradparam_LT, gradword_LT = sample.treepair[0].grad(self.Param, gradLV)
        gradparam_RT, gradword_RT = sample.treepair[1].grad(self.Param, gradRV)
        gradparam = Param(gradparam_LT.L + gradparam_RT.L,
                          gradparam_LT.R + gradparam_RT.R,
                          gradparam_LT.bias + gradparam_RT.bias)
        # Merge word grad:
        gradword = {}
        for item in [gradword_LT, gradword_RT]:
            for (word, val) in item.iteritems():
                try:
                    val = gradword[word]
                    gradword[word] += val
                except KeyError:
                    gradword[word] = val
        return {"U" : gradU, "b" : gradb, "Param" : gradparam, "Word":gradword}
    

    def predict(self, sample, dropout=True):
        """ Discourse relation prediction on the given sample
        """
        # Combine left/right vec for classification
        pred_prob, left_vec, right_vec = self.combine(sample, dropout)
        return pred_prob, left_vec, right_vec



