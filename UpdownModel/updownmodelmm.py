## updownmodelmm.py
## Author: Yangfeng Ji
## Date: 10-07-2014
## Time-stamp: <yangfeng 10/29/2014 14:39:21>

from addpath import checkpath
checkpath()

import numpy, gzip, sys
from cPickle import load, dump
from UpwardModel.util import gradclip, softmax
from updownmodel import UpdownModel


class UpdownModelMM(UpdownModel):
    def __init__(self, nDim, nRela, rng):
        """ Initialization
        """
        UpdownModel.__init__(self, nDim, nRela, rng)


    def objective(self, prob, ridx):
        """ Max-margin objective function
        """
        obj = 0.0
        for idx in range(self.nRela):
            if idx != ridx:
                obj += max(0, 1 - prob[ridx] + prob[idx])
        return obj


    def predict(self, sample):
        """ Predict on the given sample
        """
        prob, vecdict, sample = self.combine(sample)
        return prob, vecdict, sample


    def grad(self, sample):
        """ Grad wrt max-margin objective function
        """
        prob, vecdict, sample = self.predict(sample)
        gradce = numpy.zeros(prob.shape)
        ridx = sample.ridx
        # Assign value to gradce
        for idx in range(self.nRela):
            if (idx != ridx) and ((prob[ridx]-prob[idx])<1.0):
                # print prob[ridx]-prob[idx]
                gradce[idx] = 1.0
        gradce /= (self.nRela-1)
        gradce[ridx] = - gradce.sum()
        # print gradce
        gradall = self.grad_given_ce(sample, gradce, vecdict)
        return gradall
