## addpath.py
## Author: Yangfeng Ji
## Date: 09-18-2014
## Time-stamp: <yangfeng 09/18/2014 10:52:42>


def checkpath(path="../"):
    import sys
    if path not in sys.path:
        sys.path.append(path)
