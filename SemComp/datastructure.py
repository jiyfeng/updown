## datastructure.py
## Author: Yangfeng Ji
## Date: 08-21-2014
## Time-stamp: <yangfeng 10/02/2014 11:54:14>

import numpy
import sys



class Logistc(object):
    """ Logistic function
    """
    def __init__(self):
        """
        """
        pass

    def eval(self, x):
        """ Evaluate
        """
        val = 1.0 / (1.0 + numpy.exp(-x))
        return val

    def grad(self, x):
        """
        """
        val = self.eval(x)
        g = val * ( 1 - val )
        return g


class Tanh(object):
    """ Tanh function, specifically the function
        defined in Efficient Backprop
    """
    def __init__(self, alpha=1e-5):
        self.alpha = alpha

    def eval(self, x):
        x = (2.0/3) * x
        exp2x = numpy.exp(-2*x)
        val = (1.7159 * (1 - exp2x) / (1 + exp2x)) + (self.alpha * x.sum())
        return val

    def grad(self, x):
        val = self.eval(x)
        g =  (1.7159 * (2.0 / 3) * (1 - (val ** 2))) + (numpy.ones(x.shape) * self.alpha)
        return g


class HardTanh(object):
    def __init__(self, thresh=1.0):
        if thresh < 0.0:
            raise ValueError("In HardTanh: thresh should be positive")
        self.thresh = thresh

    def eval(self, x):
        hidx = numpy.where(x > self.thresh)[0]
        lidx = numpy.where(x < -self.thresh)[0]
        x[lidx] = 0
        x[lidx] = 0
        return x

    def grad(self, x):
        g = numpy.ones(x.shape)
        hidx = numpy.where(x > self.thresh)[0]
        lidx = numpy.where(x < -self.thresh)[0]
        g[hidx] = 0
        g[lidx] = 0
        return g
    
    
class Operator(Tanh):
    """ Composition operator aka the nonlinear mapping function

    In this case, it is tanh function
    """
    def __init__(self):
        """
        """
        Tanh.__init__(self)


# ----------------------------------------------------------
class Param(object):
    def __init__(self, L, R, bias):
        """ Parameters for composition

        :type L: 2-d numpy.array
        :param L: composition matrix for left node

        :type R: 2-d numpy.array
        :param R: composition matrix for right node

        :type bias: 1-d numpy.array
        :param bias: composition bias
        """
        self.L = L
        self.R = R
        self.bias = bias


# ----------------------------------------------------------
class Sample(object):
    def __init__(self, ridx, treepair, featvec=None,
                 dtreepairs=None):
        """ Initialize parameters

        :type ridx: int
        :param ridx: discourse relation index

        :type uptreepair: a tuple of Tree class instances
        :param uptreepair: upward composition tree

        :type featvec: 1-D numpy.array
        :param featvec: numeric vector of surface level features
        """
        self.ridx = ridx
        self.treepair = treepair # (left-tree, right-tree)
        self.featvec = featvec
        self.dtreepairs = dtreepairs # For downward version


# ----------------------------------------------------------
class ProdRule(object):
    def __init__(self, H=None, L=None, R=None):
        """ Production rule: H -> L R
            We don't represent unary relation -YJ
        
        :type H: string
        :param H: head node

        :type L: string
        :param L: left children node

        :type R: string
        :param R: right children node
        """
        self.H = H
        self.L = L
        self.R = R


class ParseError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

    
class BuildTreeError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Logger(object):
    def __init__(self, filename="output.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
