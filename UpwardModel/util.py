## util.py
## Author: Yangfeng Ji
## Date: 08-27-2014
## Time-stamp: <yangfeng 09/20/2014 15:35:15>

""" Util function for this project
"""

import psutil, numpy
from random import shuffle
from scipy.misc import logsumexp

def memory_usage():
    """ Check the memory usage
    """
    process = psutil.Process(os.getpid())
    mem = process.get_memory_info()[0] / float(2**20)
    sample = None
    print 'Memory usage: {} MB'.format(mem)


def balance_sample(Samples):
    """ 
    """
    dist = {}
    NewSamples = []
    for sample in Samples:
        try:
            lst = dist[sample['ridx']]
            dist[sample['ridx']].append(sample)
        except KeyError:
            dist[sample['ridx']] = [sample]
    # 
    maxval = 0
    for (key, lst) in dist.iteritems():
        if maxval < len(dist[key]):
            maxval = len(dist[key])
    # Extend list
    for (key, lst) in dist.iteritems():
        new_list = []
        while len(new_list) < maxval:
            new_list += lst
        NewSamples += new_list[:maxval]
    shuffle(NewSamples)
    return NewSamples
        
    
def softmax(x):
    """ Softmax function
    """
    Z = logsumexp(x)
    prob = numpy.exp(x - Z)
    return prob


def gradclip(g, normval, thresh):
    """ Clipping gradient with a given threshold
    """
    if (thresh is not None) and (normval > thresh):
        g = (g * thresh) / normval
    return g
