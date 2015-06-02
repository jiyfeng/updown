## node.py
## Author: Yangfeng Ji
## Date: 08-21-2014
## Time-stamp: <yangfeng 02/11/2015 10:13:07>

import numpy
from datastructure import *


class Node(object):
    def __init__(self, pnode=None, snode=None, value=None,
                 widx=None, pathcode=None):
        """ Initialize a tree node

        :type pnode: instance of Node class
        :param pnode: the parent node

        :type value: 1-d numpy.array
        :param value: value of this node (during construction,
                      it only works for leaf nodes)

        :type widx: int
        :param widx: for leaf node, it's the index of word
                     representation in the whole word matrix

        :type pathcode: binary string
        :param pathcode: nothing but indicating the position
                         of this node
        """
        # Token/POS tag
        self.token, self.pos = None, None
        # Parent/Sibling node
        self.pnode, self.snode = pnode, snode
        # Children nodes
        self.lnode, self.rnode = None, None
        # Value/Wordindex(only works for leaf node)
        self.value, self.widx = value, widx
        # Gradient information back from parent node,
        # wrt the input from this node to the parent
        # node
        self.grad_parent = None
        # Self-gradient/Parameter gradient
        self.grad, self.grad_param = None, None
        # Composition operator/Production rule
        self.opt, self.pr = Operator(), ProdRule()
        # Pathcode/Coref index
        self.pathcode, self.crindex = pathcode, None
        # Label for upward/downward model, which can be
        #   used to select the right param
        self.label = 'up' # 'up' or 'down'


    def is_leaf(self):
        """ Whether this node is a leaf node
        """
        if (self.lnode is None) and (self.rnode is None):
            return True
        else:
            return False
        
        
    def comp(self, param):
        """ Composition function
            For internal node, compute the composition function
            For leaf node, return the value directly

        :type param: instance of Param class or dict of instances
        :param param: composition matrices and bias
        """
        # Select the right param
        if type(param) == dict:
            # print 'label = ', self.label
            tparam = param[self.label] # Select param
        else:
            tparam = param
        if (self.lnode is None) and (self.rnode is None):
            # If both children nodes are None
            return self.value
        elif (self.lnode is not None) and (self.rnode is not None):
            try:
                u = tparam.L.dot(self.lnode.value)
                u += tparam.R.dot(self.rnode.value) 
            except TypeError:
                print self.lnode.pos, self.lnode.value
                print self.rnode.pos, self.rnode.value
                import sys
                sys.exit()
            self.value = self.opt.eval(u)
        else:
            raise ValueError("Unrecognized situation")
            
    def paramgrad(self, param):
        """ Compute the param gradient given current param,
            which is used to update param

        :type param: instance of Param class
        :param param: composition metrices and bias

        :type grad_parent: 1-d numpy.array
        :param grad_parent: gradient information from parent node
        """
        # Select the right param
        if type(param) == dict:
            param = param[self.label]
        # Element-wise multiplication, 1-d numpy.array
        gu = self.opt.grad(self.value) * self.grad_parent
        if self.lnode is None:
            gL = numpy.zeros(param.L.shape)
        else:
            gL = numpy.outer(gu, self.lnode.value)
        if self.rnode is None:
            gR = numpy.zeros(param.R.shape)
        else:
            gR = numpy.outer(gu, self.rnode.value)
        if (self.rnode is None) and (self.lnode is None):
            gbias = numpy.zeros(param.bias.shape)
        else:
            # Without bias term in composition
            gbias = numpy.zeros(param.bias.shape)
        # print gL.shape, gR.shape, gbias.shape
        self.grad_param = Param(gL, gR, gbias)


    def grad_input(self, param):
        """ Compute the gradient wrt input from left child node,
            and right child node. If it is a leaf node, return
            the gradient information from parent node directly.
            This is mainly for back-propagating gradient
            information

        :type grad_parent: 1-d numpy.array
        :param grad_parent: gradient information from parent node
        """
        # Select the right param
        if type(param) == dict:
            param = param[self.label]
        # Compute gradient
        if (self.lnode is None) and (self.rnode is None):
            self.grad = self.grad_parent
            return self.grad_parent
        else:
            # Is it safe to use the value directly?
            # What if it is not updated with the new param?
            gu = self.opt.grad(self.value) * self.grad_parent
            # Assign gradient back to children nodes
            # For left node
            grad_l = numpy.dot(param.L.T, gu)
            self.lnode.grad_parent = grad_l
            # For right node
            grad_r = numpy.dot(param.R.T, gu)
            self.rnode.grad_parent = grad_r
            return (grad_l, grad_r)
        # raise ValueError("Need double check")


if __name__ == '__main__':
    node = Node()
