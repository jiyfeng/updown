## util.py
## Author: Yangfeng Ji
## Date: 10-02-2014
## Time-stamp: <yangfeng 10/02/2014 14:42:14>

def preorder_bracket(root, strtree):
    if root is None:
        return strtree
    elif root.is_leaf():
        # print root.pos, root.label, root.token, root.is_leaf()
        strtree = strtree + "(" + root.pos + "/" + root.label + " " + root.token
    elif (not root.is_leaf()):
        # print root.pos, root.label, root.token, root.is_leaf()
        strtree = strtree + "(" + root.pos + "/" + root.label
    else:
        raise ValueError("Unrecognized situation in preorder bracket")
    strtree = preorder_bracket(root.lnode, strtree)
    strtree = preorder_bracket(root.rnode, strtree)
    strtree += ")"
    return strtree
        
