## util.py
## Author: Yangfeng Ji
## Date: 09-30-2014
## Time-stamp: <yangfeng 10/02/2014 16:14:42>

from SemComp.datastructure import Param
import numpy

def initparam(nDim, rng, initval=None):
    """ Initialize composition parameters

    :type nDim: int
    :param nDim: latent dimension

    :type rng: numpy.random.RandomState
    :param rng: random seed

    :type initval: float
    :param initval: initial value,
                    if None, then random value
    """
    L = numpy.asarray(rng.uniform(
        low=-numpy.sqrt(6. / (nDim + nDim)),
        high=numpy.sqrt(6. / (nDim + nDim)),
        size=(nDim, nDim)))
    if initval is not None:
        L = (L * 0.0) + initval
    R = numpy.asarray(rng.uniform(
        low=-numpy.sqrt(6. / (nDim + nDim)),
        high=numpy.sqrt(6. / (nDim + nDim)),
        size=(nDim, nDim)))
    if initval is not None:
        R = (R * 0.0) + initval
    bias = numpy.zeros((nDim,))
    if initval is not None:
        bias = (bias * 0.0) + initval
    param = Param(L, R, bias)
    return param


def initU_2v(nDim, nRela, rng, initval=None):
    """ Initialize parameters for vector product
        classification model:
        U^T * [vl,vr]

    :type nDim: int
    :param nDim: latent dimension
        
    :type nRela: int
    :param nRela: number of discourse relations

    :type rng: numpy.random.RandomState
    :param rng: random seed

    :type initval: float
    :param initval: initial value
    """
    U = {}
    for nr in range(nRela):
        Unr = numpy.asarray(rng.uniform(
            low=-1e-5, high=1e-5,
            size=(2*nDim,)))
        if initval is not None:
            Unr = (Unr * 0.0) + initval
        U[nr] = Unr
    return U
