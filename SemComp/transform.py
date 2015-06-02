## transform.py
## Author: Yangfeng Ji
## Date: 08-25-2014
## Time-stamp: <yangfeng 11/12/2014 20:44:30>

from datastructure import *
from node import *
from tree import *
import copy


class Transform(object):
    def __init__(self):
        """ Transform the original tree to be a new tree with
            the given node as root
        """
        pass


    def build(self, node, root):
        """ Build a composition tree with downward composition
            parameters

        :type node: instance of Node class
        :param node: root node of the new tree

        :type root: instance of Node class
        :param root: root node of the original tree
        """
        # raise ValueError("Check the build() function from Transform class carefully!")
        # Create root node of this new tree
        newnode = self.copynode(node, root, label='down')
        # Build the new tree
        if node.snode is not None:
            newnode.rnode = self.copysubtree(node.snode, root)
        if node.pnode is not None:
            newnode.lnode = self.build(node.pnode, root)
        # Return the root node
        return newnode


    def copynode(self, node, root, label):
        """ Copy all properties from node to a new node

        :type node: instance of Node class
        :param node: a node from the original tree

        :type label: string
        :param label: indicating where this node comes from
        """
        newnode = Node()
        newnode.label = label
        if label == 'up':
            newnode.value = copy.copy(node.value)
            newnode.pos = copy.copy(node.pos)
            newnode.token = copy.copy(node.token)
        elif label == 'down':
            newnode.pos = copy.copy(node.pos)
            if node == root:
                # print 'Find root node in tree transformation'
                newnode.token = ""
                # newnode.value = node.value
                newnode.value = numpy.zeros(node.value.shape)
            else:
                newnode.token = copy.copy(node.token)
        else:
            raise ValueError("Unrecognized label in copy node")
        return newnode


    def copysubtree(self, node, root):
        """ Copy the subtree with this node as root
        """
        root = self.copynode(node, root, 'up')
        if node.lnode is not None:
            root.lnode = self.copysubtree(node.lnode, root)
        if node.rnode is not None:
            root.rnode = self.copysubtree(node.rnode, root)
        return root
        
        
