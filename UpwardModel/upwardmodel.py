## model.py
## Author: Yangfeng Ji
## Date: 08-21-2014
## Time-stamp: <yangfeng 11/07/2014 22:59:44>

"""
Build the classification model for discourse
relation identification.
Refer to the note for more technical details
--- YJ
"""
from addpath import checkpath
checkpath()

import numpy, gzip, sys
from numpy.linalg import norm
from SemComp.tree import *
from util import *
from cPickle import load, dump


class UpwardModel(object):
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
        self.U = {}
        for nrela in range(nRela):
            U = numpy.asarray(rng.uniform(
                low=-1e-5, high=1e-5,
                size=(nDim, nDim)))
            self.U[nrela] = U 
        self.b = numpy.zeros((nRela,))
        # Parameters for upward composition
        L = numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / (nDim + nDim)),
            high=numpy.sqrt(6. / (nDim + nDim)),
            size=(nDim, nDim)))
        R = numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / (nDim + nDim)),
            high=numpy.sqrt(6. / (nDim + nDim)),
            size=(nDim, nDim)))
        bias = numpy.zeros((nDim,))
        self.Param = Param(L, R, bias)
        # For AdaGrad
        self.USqSum = {}
        for nrela in range(nRela):
            self.USqSum[nrela] = numpy.zeros(self.U[nrela].shape)
        self.bSqSum = numpy.zeros(self.b.shape)
        L = numpy.zeros(L.shape)
        R = numpy.zeros(R.shape)
        self.ParamSqSum = Param(L, R, bias)


    def cross_entropy(self, prob, ridx):
        """ Cross entropy of the given sample
        Minimize the cross entropy to update the parameters

        :type sample: instance of Sample class
        :param sample: One sample
        """
        return -numpy.log(prob[ridx])


    def grad(self, sample):
        """ Gradient of parameters wrt the given sample

        Get the gradient of vocab also from this function,
        but doesn't implement the update in this class

        :type sample: instance of Sample class
        :param sample: see datastructure for more info
        """
        # Grad of CE (cross entropy) wrt softmax input x
        prob, vl, vr = self.predict(sample) # Output prob
        tgt = numpy.zeros(prob.shape)
        tgt[sample.ridx] = 1.0
        gradCE_x = prob - tgt # (nRela,)
        gradall = self.grad_given_ce(sample, gradCE_x, vl, vr)
        return gradall
    

    def grad_given_ce(self, sample, gradCE_x, vl, vr):
        """ Grad of params given the grad of cross entropy
            The purpose of spliting this part from grad()
            is to make it reusable for joint model

            The basic idea is to get rid of prediction function
        """
        # Grad of CE wrt U_r
        gradU = {}
        for c in range(len(self.U)):
            gradU[c] = gradCE_x[c] * numpy.outer(vl, vr)
        # Grad of CE wrt b
        gradb = gradCE_x
        # Grad of CE wrt left/right vector
        gradLV = numpy.zeros(vl.shape)
        gradRV = numpy.zeros(vr.shape)
        for c in range(len(self.U)):
            gradLV += gradCE_x[c] * self.U[c].dot(vr)
            gradRV += gradCE_x[c] * self.U[c].T.dot(vl)
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
        return {"U" : gradU, "b" : gradb, "Param" : gradparam,
                "Word":gradword}


    def grad_minibatch(self, sample_list):
        """ Compute gradient from mini-batch data size

        :type sample_list: list
        :param sample_list: a list of sample list
        """
        for (idx, sample) in enumerate(sample_list):
            if idx == 0:
                gradparamsum = self.grad(sample)
            else:
                gradparam = self.grad(sample)
                for (key, gradcomp) in gradparam.iteritems():
                    if key == "b":
                        gradparamsum["b"] += gradcomp
                    elif key == "U":
                        for c in range(len(gradcomp)):
                            gradparamsum["U"][c] += gradcomp[c]
                    elif key == "Param":
                        gradparamsum["Param"].L += gradcomp.L
                        gradparamsum["Param"].R += gradcomp.R
                        gradparamsum["Param"].bias += gradcomp.bias
                    elif key == 'Word':
                        for (word, g) in gradcomp.iteritems():
                            try:
                                # Word already here
                                gradparamsum['Word'][word] += g
                            except KeyError:
                                # First time see word
                                gradparamsum['Word'][word] = g
                    else:
                        raise KeyError("Unmatched key value: {}".format(key))
        # Average
        N = len(sample_list)
        for (key, gradcomp) in gradparamsum.iteritems():
            if key == "b":
                gradparamsum["b"] /= N
            elif key == "U":
                for c in range(len(gradcomp)):
                    gradparamsum["U"][c] /= N
            elif key == "Param":
                gradparamsum["Param"].L /= N
                gradparamsum["Param"].R /= N
                gradparamsum["Param"].bias /= N
            elif key == 'Word':
                for word in gradcomp.iterkeys():
                    gradparamsum['Word'][word] /= N
            else:
                raise KeyError("Unmatched key value: {}".format(key))
        return gradparamsum
    

    def update_adagrad(self, grad, lr, reg, updateparam=True,
                       updatecomp=True, norm_thresh=None):
        """ Update parameters with AdaGrad
        """
        normsq = 0.0
        lrclass, lrcomp = lr['class'], lr['comp']
        regup, regclass = reg['up'], reg['param']
        for c in range(len(grad["U"])):
            normsq += (grad["U"][c]**2).sum()
        normsq += norm(grad["b"]**2).sum()
        normsq += norm(grad["Param"].L**2).sum()
        normsq += norm(grad["Param"].R**2).sum()
        # normsq += norm(grad["Param"].bias**2).sum()
        normval = numpy.sqrt(normsq)
        if (norm_thresh is not None) and (normval > norm_thresh):
            print '\tnormval = {0:.3f}'.format(normval)
            if normval > (2000 * norm_thresh):
                print 'Too big normval, exit program'
                sys.exit()
        if updateparam:
            # Classification parameter
            # print 'Update classification parameter'
            for c in range(len(self.U)):
                g = gradclip(grad['U'][c], normval, norm_thresh)
                self.USqSum[c] += g ** 2
                g /= numpy.sqrt(self.USqSum[c])
                self.U[c] = self.U[c] - (lrclass * (g + regclass * self.U[c]))
            # Classification bias
            g = gradclip(grad['b'], normval, norm_thresh)
            self.bSqSum += g ** 2
            g /= numpy.sqrt(self.bSqSum)
            self.b = self.b - (lrclass * (g + regclass * self.b))
        if updatecomp:
            # Composition parameter
            # Left
            g = gradclip(grad['Param'].L, normval, norm_thresh)
            self.ParamSqSum.L += g ** 2
            g /= numpy.sqrt(self.ParamSqSum.L)
            self.Param.L = self.Param.L - (lrcomp * (g + regup * self.Param.L))
            # Right
            g = gradclip(grad['Param'].R, normval, norm_thresh)
            self.ParamSqSum.R += g ** 2
            g /= numpy.sqrt(self.ParamSqSum.R)
            self.Param.R = self.Param.R - (lrcomp * (g + regup * self.Param.R))
        

    def predict(self, sample, dropout=True):
        """ Discourse relation prediction on the given sample
            prob = softmax(v_l * U * v_r + b)
        """
        # Combine left/right vec for classification
        pred_prob, left_vec, right_vec = self.combine(sample,
                                                      dropout)
        # Compute softmax for probability
        # print sample.ridx, pred_prob, left_vec, right_vec
        pred_prob = softmax(pred_prob)
        # Output: Predicted prob, left composition vector
        #         right composition vector
        return pred_prob, left_vec, right_vec


    def combine(self, sample, dropout=False):
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
            prob[idx] = left_vec.T.dot(self.U[idx].dot(right_vec))
        prob += self.b
        return prob, left_vec, right_vec


    def getparams(self):
        """ Get all parameters in this model
        """
        return {"U":self.U, "b":self.b, "Param":self.Param}


    def assignparams(self, D):
        """ Assign values to params

        :type D: dict
        :param D: has the same form as in getparams()
        """
        try:
            self.U = D['U']
            self.b = D['b']
            self.Param = D['Param']
        except KeyError:
            print 'In UpwardModel: Failure on assigning values to params'
    
    
    def savemodel(self, fname):
        """ Save model into fname
        """
        if not fname.endswith('.pickle.gz'):
            fname = fname + '.pickle.gz'
        D = self.getparams()
        with gzip.open(fname, 'w') as fout:
            dump(D, fout)
        print 'Save model into file {}'.format(fname)


    def loadmodel(self, fname):
        """ Load model from fname
        """
        with gzip.open(fname, 'r') as fin:
            D = load(fin)
        self.assignparams(D)
        print 'Load model from file: {}'.format(fname)
