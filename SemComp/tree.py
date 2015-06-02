## tree.py
## Author: Yangfeng Ji
## Date: 08-21-2014
## Time-stamp: <yangfeng 02/11/2015 10:15:09>

""" Semantic composition based binary tree structure
"""

from datastructure import *
from node import *
from buildtree import *
from util import preorder_bracket

class Tree(object):
    """ Tree structure for upward semantic model
    """
    def __init__(self):
        """ Initialize tree structure

        :type param: instance of Param class
        :param param: parameter for upward semantic
                      composition

        :type backparam: instance of Param class
        :param backparam: parameter for downward semantic
                          composition
        """
        self.root = None
        self.nodelist = None
        self.entity_nodedict = None
        

    def build(self, strTree, vocab):
        """ (1) Read structure from file, and
            (2) build tree structure, and
            (3) initialize leaf node

        Refer buildtree.py for more detail

        :type strTree: string
        :param strTree: string of the tree information

        :type vocab: Vocab instance
        :param vocab: word vocab with representation
        """
        self.root = buildtree(strTree, vocab)
        

    def upward(self, param):
        """ Upward semantic composition step ---
            Using forward to compute the composition result
            of the root node

        Find a topological order of all nodes, then follow
        the linear order to update the node value one by one
        """
        # If the node list is empty, do BFT first
        if self.nodelist is None:
            self.bft()
        # Reverse the nodelist, without change the original one
        self.nodelist.reverse() # Reverse
        for node in self.nodelist:
            if node.is_leaf(): # Leaf node
                pass
            else: # Non-leaf node
                node.comp(param)
        # Return the composition result
        self.nodelist.reverse() # Reverse back
        return self.root.value
    

    def grad(self, param, grad_input):
        """ Using back-propagation to compute upward gradients
            wrt parameters

        :type grad_input: 1-d numpy.array
        :param grad_input: gradient information from model
        (In other words, this is the gradient of the model
        wrt to the composition result)
        """
        # Assign the grad_input
        self.root.grad_parent = grad_input
        # If the node list is empty, do BFT first
        if self.nodelist is None:
            self.bft()
        # Back propagating the gradient from root to leaf
        for node in self.nodelist:
            # For model parameters
            node.paramgrad(param)
            # For back propagation
            node.grad_input(param)
        # Collecting grad information
        # Traversing all the interval nodes and adding
        # all the grad information wrt param together
        if type(param) == dict:
            # Depends on the type of param
            gparam = {}
            for key in param.iterkeys():
                gparam[key] = Param(numpy.zeros(param[key].L.shape),
                    numpy.zeros(param[key].R.shape),
                    numpy.zeros(param[key].bias.shape))
        else:
            gparam = Param(numpy.zeros(param.L.shape),
                numpy.zeros(param.R.shape),
                numpy.zeros(param.bias.shape))
        # Gradient of words
        gword = {}
        # print '-----------------------------------'
        for node in self.nodelist:
            # What will happen if the node is leaf node?
            # Already solved by setting the grad to be zero
            # Check the paramgrad in Node class
            if type(gparam) == dict:
                key = node.label
                gparam[key].L += node.grad_param.L
                gparam[key].R += node.grad_param.R
            else:
                gparam.L += node.grad_param.L
                gparam.R += node.grad_param.R
            if node.is_leaf():
                gword[node.token] = node.grad
        # print numpy.linalg.norm(gparam['up'].L)
        return gparam, gword
    

    def update(self):
        """ Update param with gradient information
        """
        raise NotImplementedError("Call the update() function in model class for updating all parameters")


    def bft(self):
        """ Breadth-first traversal on the tree (starting
            from root note!)
        """
        self.nodelist = []
        if self.root is not None:
            queue = [self.root]
            while queue:
                node = queue.pop(0)
                if node.lnode is not None:
                    queue.append(node.lnode)
                if node.rnode is not None:
                    queue.append(node.rnode)
                self.nodelist.append(node)


    def find_entitynodes(self):
        """ Go through all the nodes to find the entity nodes
        """
        if self.nodelist is None:
            self.bft()
        self.entity_nodedict = {}
        for node in self.nodelist:
            if node.crindex is not None:
                self.entity_nodedict[node.crindex] = node


    def bracketing(self):
        """ Generating bracketing results about this tree
        """
        strtree = preorder_bracket(self.root, "")
        strtree = "(" + strtree + ")"
        return strtree
