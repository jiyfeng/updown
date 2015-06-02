## vocab.py
## Author: Yangfeng Ji
## Date: 09-22-2014
## Time-stamp: <yangfeng 09/23/2014 17:18:17>

""" The Vocab class, which includes:
1, Load vocab from other information resources,
   for example word2vec
2, Create an vocab based the word list, while each
   word representation is a random vector
3, SGD update with the given gradient information (AdaGrad)
4, Get the vector representation for a given word
5, Get the index for a given word
--- YJ
"""

import numpy, sys

rng = numpy.random.RandomState(1234)

class Vocab(object):
    def __init__(self, dct, W=None):
        """ Initialization of a vocab

        :type dct: dictionary with {string:int}
        :param dct: key to key index mapping

        :type W: numpy.array
        :param W: numeric representation of strings in dct 
        """
        self.dict = dct
        self.W = W
        self.nWord = len(dct)
        self.nDim = None
        self.WSqSum = numpy.zeros(self.W.shape)


    def init(self, nDim, std=1.0):
        """ Randomly initialize word representation

        :type nDim: int
        :param nDim: latent dimension of word rep
        """
        self.nDim = nDim
        self.W = numpy.asarray(rng.uniform(
            high=0.001, low=-0.001,
            size=(self.nWord, self.nDim)))
        

    def getvec(self, s):
        """ Get vector representation for a given word
        """
        if self.nDim is None:
            self.nDim = self.W.shape[0]
        try:
            return self.W[:,self.dict[s]]
        except KeyError:
            # print 'Could not find vector rep of {},
            #   return with zero vector'.format(s)
            return numpy.zeros((self.nDim,))
            # return numpy.random.random((self.D,))/1000

            
    def getLatDim(self):
        """ Get latent dimensions
        """
        return self.W.shape[0]


    def getWidx(self, word):
        """ Get word index

        :type word: string
        :param word: 
        """
        try:
            idx = self.dict[word]
            return idx
        except KeyError:
            return None


    def update_adagrad(self, grad, lr, reg, norm_thresh=None):
        """ Using AdaGrad to update word representation
        """
        # print 'word grad = {}'.format(grad)
        # print numpy.sqrt((self.WSqSum**2).sum())
        for (word, g) in grad.iteritems():
            widx = self.getWidx(word)
            if widx is not None:
                self.WSqSum[:,widx] += g**2
                g /= numpy.sqrt(self.WSqSum[:,widx])
                self.W[:,widx] = self.W[:,widx] - (lr * (g + reg * self.W[:,widx]))
        

if __name__ == '__main__':
    dct = {'hello':0, 'world':1}
    vocab = Vocab(dct)
    vocab.init(10, std=1.0)
    print vocab.W
