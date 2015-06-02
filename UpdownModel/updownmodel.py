## updownmodel.py
## Author: Yangfeng Ji
## Date: 09-30-2014
## Time-stamp: <yangfeng 02/16/2015 12:25:52>

""" Build a up-down semantic model for discourse
    relation identification
    - YJ
"""

from addpath import checkpath
checkpath()

from UpwardModel.util import gradclip, softmax
from SemComp.transform import Transform
from SemComp.tree import Tree
from util import initparam, initU_2v
from cPickle import load, dump
import numpy, gzip, sys, copy


class UpdownModel(object):
    def __init__(self, nDim, nRela, rng):
        """ Initialization

        :type nDim: int
        :param nDim: latent dimension

        :type nRela: int
        :param nRela: number of discourse relations

        :type rng: numpy.random.RandomState
        :param rng: random seed for initializing params
        """
        self.updateup, self.updatedown = True, True
        self.updateparam = True
        self.nDim, self.nRela, self.rng = nDim, nRela, rng
        # Classification layer: UpU, DownU, b
        UpU = initU_2v(nDim, nRela, rng)
        DownU = initU_2v(nDim, nRela, rng)
        b = numpy.zeros((nRela,))
        # Composition layer: UpParam, DownParam
        UpParam = initparam(nDim, rng)
        DownParam = initparam(nDim, rng)
        Param = {'up':UpParam, 'down':DownParam}
        self.ParamAll = {'UpU':UpU, 'DownU':DownU, 'b':b,
            'Param':Param}
        # For agagrad
        self.SqSum = {}
        # classification layer: UpUSqSum, DownUSqSum, bSqSum
        UpUSqSum = initU_2v(nDim, nRela, rng, 1.0)
        DownUSqSum = initU_2v(nDim, nRela, rng, 1.0)
        bSqSum = numpy.zeros((nRela,))
        # composition layer
        UpParamSqSum = initparam(nDim, rng, 1.0)
        DownParamSqSum = initparam(nDim, rng, 1.0)
        ParamSqSum = {'up':UpParamSqSum, 'down':DownParamSqSum}
        self.SqSum = {'UpU':UpUSqSum, 'DownU':DownUSqSum,
            'b':bSqSum, 'Param':ParamSqSum}


    def objective(self, prob, ridx):
        """ Cross entropy as objective function
            (Can use others later)

        :type prob: 1-D numpy.array
        :param prob: predicted values for each discourse relation

        :type ridx: int
        :param ridx: index of true label
        """
        return -numpy.log(prob[ridx])


    def predict(self, sample):
        """ Discourse relation prediction on the given sample
        """
        predprob, vecdict, sample = self.combine(sample)
        predprob = softmax(predprob)
        return predprob, vecdict, sample


    def combine(self, sample):
        """ Combine each part for final prediction
        """
        prob, vecdict = numpy.zeros((self.nRela,)), {}
        # First, upward model
        # print '-------up part---------------------------'
        tpair = sample.treepair
        upleftvec = tpair[0].upward(self.ParamAll['Param'])
        uprightvec = tpair[1].upward(self.ParamAll['Param'])
        # Create down tree pairs
        sample = self.createdtreepairs(sample)
        # print '-------down part-------------------------'
        vec = numpy.hstack((upleftvec, uprightvec))
        for (idx, Uidx) in self.ParamAll['UpU'].iteritems():
            prob[idx] += Uidx.dot(vec)
        # print 'old_prob = {}'.format(prob)
        vecdict['up'] = (upleftvec, uprightvec)
        # Second, downward model
        downvecdict = {}
        dprob = numpy.zeros(prob.shape)
        # DOUBLE CHECK THIS PART !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for (cridx, tpair) in sample.dtreepairs.iteritems():
            # print 'cridx = {}'.format(cridx)
            downleftvec = tpair[0].upward(self.ParamAll['Param'])
            downrightvec = tpair[1].upward(self.ParamAll['Param'])
            downvecdict[cridx] = (downleftvec, downrightvec)
            vec = numpy.hstack((downleftvec, downrightvec))
            for (idx, Uidx) in self.ParamAll['DownU'].iteritems():
                dprob[idx] += Uidx.dot(vec)
        vecdict['down'] = downvecdict
        prob += dprob
        # Combine with bias term
        prob += self.ParamAll['b']
        return prob, vecdict, sample


    def createdtreepairs(self, sample, with_alignment=True):
        Lt, Rt = sample.treepair
        # We can modified this part to make every mention
        #   from Arg1 aligned with every one in Arg2
        # Find shared entity node 
        entnodepairs = {}
        if with_alignment:
            for (cridx, lnode) in Lt.entity_nodedict.iteritems():
                try:
                    rnode = Rt.entity_nodedict[cridx]
                    entnodepairs[cridx] = (lnode, rnode)
                except KeyError:
                    pass
        else:
            counter = 0
            for lnode in Lt.entity_nodedict.itervalues():
                for rnode in Rt.entity_nodedict.itervalues():
                    entnodepairs[counter] = (lnode, rnode)
                    counter += 1
        # Transform tree structure for a given entity node
        dtreepairs = {}
        tf = Transform()
        for (cridx, nodepair) in entnodepairs.iteritems():
            dltree, drtree = Tree(), Tree()
            dltree.root = tf.build(nodepair[0], Lt.root)
            drtree.root = tf.build(nodepair[1], Rt.root)
            dtreepairs[cridx] = (dltree, drtree)
        sample.dtreepairs = dtreepairs
        return sample


    def grad(self, sample):
        """ Compute grad for one given sample

        :type sample: Sample instance
        :param sample: one training sample
        """
        prob, vecdict, sample = self.predict(sample)
        # Grad from classification layer
        tgt = numpy.zeros(prob.shape)
        tgt[sample.ridx] = 1.0
        gradce = prob - tgt
        # Get grad for given sample and grad on cross-entropy
        gradall = self.grad_given_ce(sample, gradce, vecdict)
        return gradall


    def grad_given_ce(self, sample, gradce, vecdict):
        """ Grad given cross-entropy and compositional vec's
        """
        # Upward classification part
        gradUpU = initU_2v(self.nDim, self.nRela, self.rng, 0.0)
        if self.updateup:
            upvec = vecdict['up']
            for idx in self.ParamAll['UpU'].iterkeys():
                # Get grad of UpU
                gradUpU[idx] = gradce[idx] * numpy.hstack(upvec)
        # Downward classification part
        gradDownU = initU_2v(self.nDim, self.nRela, self.rng, 0.0)
        if self.updatedown:
            for downvec in vecdict['down'].itervalues():
                # Existing downward tree pairs
                for idx in self.ParamAll['DownU'].iterkeys():
                    downvec = numpy.hstack(downvec)
                    gradDownU[idx] += gradce[idx] * downvec
        # Bias in classification part
        if self.updateup:
            gradb = gradce
        else:
            gradb = numpy.zeros(gradce.shape)
        # ------------------------------------------
        # Param for the composition layer
        gradparam = {}
        for key in self.ParamAll['Param'].iterkeys():
            gradparam[key] = initparam(self.nDim, self.rng, 0.0)
        if self.updateparam:
            # Upward composition part
            tpair, vecpair = sample.treepair, vecdict['up']
            gup = self.grad_treepair(tpair, gradce, vecpair,
                self.ParamAll['UpU'], self.ParamAll['Param'])
            gradparam = self.mergeparamgrad(gradparam, gup)
            # DOUBLE CHECK THIS PART !!!!!!!!!!!!
            # Downward composition part
            for (cridx, tpair) in sample.dtreepairs.iteritems():
                # Existing downward tree pairs
                vecpair = vecdict['down'][cridx]
                gdown = self.grad_treepair(tpair, gradce, vecpair,
                    self.ParamAll['DownU'], self.ParamAll['Param'])
                # Combine with graddownparam
                gradparam = self.mergeparamgrad(gradparam, gdown)
        # No word grad information please !!!
        return {"UpU":gradUpU, "DownU":gradDownU, "b":gradb,
                "Param":gradparam}


    def mergeparamgrad(self, gp1, gp2):
        """ Add the grad information from gp2 to gp1

        :type gp1: dict
        :param gp1: a dict of param grad including both the
                    gradient on both upward operator and
                    downward operator

        :type gp2: dict
        :param gp2:
        """
        for key in gp2.iterkeys():
            try:
                gp1[key].L += gp2[key].L
                gp1[key].R += gp2[key].R
            except KeyError:
                raise ValueError("Unrecognized key in merge param grad")
        return gp1


    def grad_treepair(self, tpair, gradce, vecpair, U, param):
        """ Compute composition param grad for a given tree pair

        :type tpair: tuple
        :param tpair: one pair of tree

        :type gradce: 1-D numpy.array
        :param gradce: grad of CE wrt each discourse relation

        :type vecpair: tuple
        :param vecpair: (left_vector, right_vector)

        :type U: dict
        :param U: classification parameter

        :type Param: instance of Param class
        :param param: composition parameter
        """
        vl, vr = vecpair
        gradlv = numpy.zeros(vl.shape)
        gradrv = numpy.zeros(vr.shape)
        gradTree = {}
        gradTree['up'] = initparam(self.nDim, self.rng, 0.0)
        gradTree['down'] = initparam(self.nDim, self.rng, 0.0)
        for (idx, Uidx) in U.iteritems():
            gradlv += gradce[idx] * U[idx][:self.nDim]
            gradrv += gradce[idx] * U[idx][self.nDim:(2*self.nDim)]
        gradLTidx, _ = tpair[0].grad(param, gradlv)
        gradRTidx, _ = tpair[1].grad(param, gradrv)
        gradTree = self.mergeparamgrad(gradTree, gradLTidx)
        gradTree = self.mergeparamgrad(gradTree, gradRTidx)
        return gradTree
        


    def grad_minibatch(self, samplelist):
        """ Compute gradient from a minibatch dataset

        :type sampelist: list
        :param samplelist: a list of sample
        """
        # Initiailzation
        Param = {}
        Param['up'] = initparam(self.nDim, self.rng, 0.0)
        Param['down'] = initparam(self.nDim, self.rng, 0.0)
        UpU = initU_2v(self.nDim, self.nRela, self.rng, 0.0)
        DownU = initU_2v(self.nDim, self.nRela, self.rng, 0.0)
        b = numpy.zeros((self.nRela,))
        gradall = {'Param':Param, 'UpU':UpU, 'DownU':DownU, 'b':b}
        # Merge grad information
        for sample in samplelist:
            ga = self.grad(sample)
            gradall = self.mergegrad(gradall, ga)
        gradall = self.averagegrad(gradall, len(samplelist))
        return gradall



    def mergegrad(self, gradsum, gradone):
        """ Merge ga into gradall

        :type gradall: dict
        :param gradall: grad for all example

        :type ga: dict
        :param ga: grad from one example
        """
        for (key, val) in gradone.iteritems():
            if (key=='UpU') or (key=='DownU'):
                for (i, Ui) in gradone[key].iteritems():
                    gradsum[key][i] += Ui
            elif (key=="b"):
                gradsum[key] += gradone[key]
            elif (key=="Param"):
                for label in gradone[key].iterkeys():
                    gradsum[key][label].L += gradone[key][label].L
                    gradsum[key][label].R += gradone[key][label].R
            else:
                raise KeyError("Unrecognized key in grad merge")
        return gradsum



    def averagegrad(self, gradall, n):
        """ Taking average on grad

        :type gradall: dict
        :param gradall: grad from all examples

        :type n: int
        :param n: number of examples
        """
        for (key, val) in gradall.iteritems():
            if (key=='UpU') or (key=='DownU'):
                for (i, Ui) in gradall[key].iteritems():
                    gradall[key][i] /= n
            elif (key=="b"):
                gradall[key] /= n
            elif (key=="Param"):
                for label in gradall[key].iterkeys():
                    gradall[key][label].L /= n
                    gradall[key][label].R /= n
            else:
                raise KeyError("Unrecognized key in grad average")
        return gradall
                


    def update_adagrad(self, gradall, lr, reg, normthresh=1.0):
        """ Update params with AdaGrad

        :type upgrad: dict
        :param upgrad: grad for upward part

        :type downgrad: dict
        :param downgrad: grad for downward part

        :type lr: float
        :param lr: learning rate

        :type reg: float
        :param reg: regularization parameter

        :type normthresh: float
        :param normthresh: threshold for grad normalization
        """
        regup, regdown, regparam = reg['up'], reg['down'], reg['param']
        lrclass, lrcomp = lr['class'], lr['comp']
        normval = self.getnormval(gradall)
        # print normthresh, normval
        if normthresh < normval:
            print '\tnormval = {0:.3f}'.format(normval)
            if normval > (2000 * normthresh):
                print 'Too big normval, exit program'
                sys.exit()
        for (key, val) in gradall.iteritems():
            if (key=="UpU"):
                if self.updateup:
                    for (i, gUi) in gradall[key].iteritems():
                        g = gradclip(gUi, normval, normthresh)
                        self.SqSum[key][i] += (g**2)
                        g = g / (numpy.sqrt(self.SqSum[key][i]) + 1e-7)
                        self.ParamAll[key][i] -= lrclass * (g + regup * self.ParamAll[key][i])
            elif (key=="DownU"):
                if self.updatedown:
                    for (i, gUi) in gradall[key].iteritems():
                        g = gradclip(gUi, normval, normthresh)
                        self.SqSum[key][i] += (g**2)
                        g = g / (numpy.sqrt(self.SqSum[key][i]) + 1e-7)
                        self.ParamAll[key][i] -= lrclass * (g + regdown * self.ParamAll[key][i])
            elif (key=="b"):
                if self.updateup:
                    g = gradclip(gradall[key], normval, normthresh)
                    self.SqSum[key] += (g**2)
                    g = g / (numpy.sqrt(self.SqSum[key]) + 1e-7)
                    # self.ParamAll[key] -= lrclass * (g + regup * self.ParamAll[key])
            elif (key=="Param"):
                if self.updateparam:
                    for label in self.ParamAll[key].iterkeys():
                        # print 'Only update downward operator'
                        # L
                        g = gradclip(gradall[key][label].L, normval, normthresh)
                        self.SqSum[key][label].L += (g ** 2)
                        g = g / (numpy.sqrt(self.SqSum[key][label].L) + 1e-7) # Avoid numeric issue
                        self.ParamAll[key][label].L -= lrcomp * (g + regparam * self.ParamAll[key][label].L)
                        # R
                        g = gradclip(gradall[key][label].R, normval, normthresh)
                        self.SqSum[key][label].R += (g ** 2)
                        g = g / (numpy.sqrt(self.SqSum[key][label].R) + 1e-7)
                        self.ParamAll[key][label].R -= lrcomp * (g + regparam * self.ParamAll[key][label].R)
            else:
                raise KeyError("Unrecoginized key in adagrad update: {}".format(key))
        
        
    def getnormval(self, gradall):
        """ Merge the grad with SqSum for AdaGrad and also
            compute the normval of grad
        """
        normval = 0.0
        for (key, val) in gradall.iteritems():
            if (key=='UpU') or (key=='DownU'):
                for (i, Ui) in gradall[key].iteritems():
                    normval += (Ui ** 2).sum()
            elif key=="b":
                normval += (gradall[key] ** 2).sum()
            elif key=="Param":
                for label in gradall[key].iterkeys():
                    normval += (gradall[key][label].L ** 2).sum()
                    normval += (gradall[key][label].R ** 2).sum()
            else:
                raise KeyError("Unrecognized key: {}".format(key))
        normval = numpy.sqrt(normval)
        return normval


    def loadmodel(self, ParamAll):
        """ Load model from ParamAll
        """
        for (key, param) in ParamAll.iteritems():
            print 'Load ParamAll[{}] from file'.format(key)
            self.ParamAll[key] = param


    def savemodel(self, fmodel):
        """ Save model into fmodel
        """
        print 'Save data into {}'.format(fmodel)
        with gzip.open(fmodel, "w") as fout:
            dump(self.ParamAll, fout)
