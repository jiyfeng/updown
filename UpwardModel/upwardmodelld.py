## model.py
## Author: Yangfeng Ji
## Date: 08-21-2014
## Time-stamp: <yangfeng 10/31/2014 21:41:55>

"""
Build the classification model for discourse
relation identification.
Refer to the note for more technical details

Still using bilinear model, but with a special
form of the bilinear matrix
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


class UpwardModelLD(UpwardModel):
    def __init__(self, nDim, nRela, rng=None):
        """ Initialize the basic model, which is the standard
            RNN based semantic composition model

        :type nDim: int
        :param nDim: latent units

        :type nRela: int
        :param nRela: number of discourse relations
        """
        if rng is None:
            rng = numpy.random.RandomState(1234)
        UpwardModel.__init__(self, nDim, nRela, rng)
        self.nDim, self.nRela = nDim, nRela
        # Overload 
        self.U = {}
        for nrela in range(nRela):
            # U = numpy.asarray(rng.uniform(
            #     low=-numpy.sqrt(6. / (nDim + nDim)),
            #     high=numpy.sqrt(6. / (nDim + nDim)),
            #     size=(nDim, nDim)))
            U = numpy.asarray(rng.uniform(
                low=-1e-5, high=1e-5, size=(nDim, 3)))
            self.U[nrela] = U 
        self.b = numpy.zeros((nRela,))
        # For AdaGrad
        self.USqSum = {}
        for nrela in range(nRela):
            self.USqSum[nrela] = numpy.ones(self.U[nrela].shape)
        self.bSqSum = numpy.ones(self.b.shape)


    def combine(self, sample, dropout=False):
        """ Combine left/right latent vec to get a un-normalized
            prob
        """
        treepair = sample.treepair
        left_vec = treepair[0].upward(self.Param)
        right_vec = treepair[1].upward(self.Param)
        # Un-normalized prob
        prob = numpy.zeros((self.nRela,))
        for nr in range(self.nRela):
            U = self.U[nr]
            prob[nr] += left_vec.dot(U[:,0])*right_vec(U[:,1])
            prob[nr] += (left_vec * right_vec).dot(U[:,2])
        prob += self.b
        return prob, left_vec, right_vec


    def grad_given_ce(self, sample, gradCE_x, vl, vr):
        """ Grad of params given the grad of cross entropy
            The purpose of spliting this part from grad()
            is to make it reusable for joint model

            The basic idea is to get rid of prediction function
        """
        # Grad of CE wrt U_r
        gradU = {}
        for c in range(self.nRela):
            U = self.U[c]
            g = numpy.zeros(U.shape)
            g[:,0] = vl
            g[:,1] = vr
            g[:,2] = vl * vr
            gradU[c] = gradCE_x[c] * g
        # Grad of CE wrt b
        gradb = gradCE_x
        # Grad of CE wrt left/right vector
        gradLV = numpy.zeros(vl.shape)
        gradRV = numpy.zeros(vr.shape)
        for (c, Uc) in self.U.iteritems():
            gradLV += gradCE_x[c] * (Uc[:,0] + (Uc[:,2] * vr))
            gradRV += gradCE_x[c] * (Uc[:,1] + (Uc[:,2] * vl))
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





