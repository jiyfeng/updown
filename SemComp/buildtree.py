## buildtree.py
## Author: Yangfeng Ji
## Date: 09-02-2014
## Time-stamp: <yangfeng 10/18/2014 12:27:42>

from node import Node, Operator
from tree import *
from datastructure import *
import sys


def getentityindex(pos):
    items = pos.split("/")
    if len(items) == 2:
        return (items[0], int(items[1]))
    else:
        return (items[0], None)
    

def buildtree(str_tree, vocab):
    """ Build tree from string str_tree

    :type str_tree: string
    :param str_tree:

    :type vocab: instance of Vocab
    :param vocab: word vocab and vector representation
    """
    str_tree = str_tree.replace("(", " ( ").replace(")", " ) ")
    neat_str = str_tree.replace("(","").replace(")","").replace(" ","")
    if len(neat_str) == 0:
        raise BuildTreeError("Empty tree")
    queue = str_tree.strip().split()
    queue.pop()
    stack = []
    while queue:
        curr_val = queue.pop(0)
        if curr_val == ")":
            if len(stack) > 1:
                try:
                    stack = operate(stack, vocab)
                except ParseError:
                    # print str_tree
                    # sys.exit()
                    raise BuildTreeError("Can not parse the tree")
            else:
                # Keep stack unchanged
                pass
        elif (curr_val != ")"):
            stack.append(curr_val)
        else:
            pass
    root = stack[-1]
    return root


def operate(stack, vocab):
    # Keep pop up, until a "("
    elemlist = []
    while stack:
        elem = stack.pop()
        if elem == "(":
            break
        else:
            elemlist.append(elem)
    try:
        if len(elemlist) == 2:
            elem1, elem2, elem3 = elemlist[0], elemlist[1], None
        elif len(elemlist) == 3:
            elem1,elem2,elem3 = elemlist[0],elemlist[1],elemlist[2]
        else:
            raise ParseError("Something wrong here")
    except TypeError:
        print 'stack = {}'.format(stack)
        print 'elemlist = {}'.format(elemlist)
        raise ParseError("Parsing error here, with \tstack = {}\nelemlist = {}".format(stack, elemlist))
    # 
    if (type(elem1) == str) and (type(elem2) == str) and (elem3 is None):
        # Leaf node
        # print 'Leaf node'
        node = Node()
        pos, crindex = getentityindex(elem2)
        # Token to be lower case
        node.token, node.pos = elem1.lower(), pos
        node.value = vocab.getvec(node.token)
        if crindex is not None:
            # This node is an entity mention
            node.crindex = crindex
    elif (type(elem1) == str) and (type(elem2) == str) and (type(elem3) == str):
        # Leaf node, where the last two are both tokens
        node = Node()
        pos, crindex = getentityindex(elem3)
        token = (elem2 + ' ' + elem1).lower()
        node.token, node.pos = token, pos
        node.value = vocab.getvec(node.token)
        if crindex is not None:
            node.crindex = crindex
    elif (isinstance(elem1, Node)) and (isinstance(elem2, Node)) and (type(elem3) == str):
        # Non-leaf node
        # print 'Non-leaf node'
        node = Node()
        node.pos, crindex = getentityindex(elem3)
        if crindex is not None:
            # This node is an entity mention
            node.crindex = crindex
        elem1.pnode, elem2.pnode = node, node
        elem1.snode, elem2.snode = elem2, elem1
        node.rnode, node.lnode = elem1, elem2
    elif (isinstance(elem1, Node)) and (type(elem2) == str) and (elem3 is None):
        # Unary relation, assign the high-level POS tag as
        #   the final POS tag
        # print 'Unary node'
        elem1.pos, crindex = getentityindex(elem2)
        if crindex is not None:
            elem1.crindex = crindex
        node = elem1
    else:
        raise ValueError("Unrecognized operation")
    stack.append(node)
    return stack
