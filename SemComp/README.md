# Code #

## Code Structure ##

- node.py: including everything can be done in a single node (1) one-step composition; (2) gradient computing wrt inputs and parameters respectively; (3) updating node value --- only for leaf node
- tree.py: including everything can be done in a single tree (1) forward propagation to compute the composition result on the root node; (2) backward propagation to compute the gradient of all parameters (besides the word representation)
- model.py: including everything about the relation identification model (1) a classifier using sofemax function; (2) gradient wrt classification parameter and inputs respectively; (3) parameter update; (4) passing gradients wrt input back to trees
- learn.py: including everything about learning parameters (1) sgd-updating; (2) learning rate updating; (3) 
- main.py: a main function to call everything here

## Preprocessing Code Structure ##

- preprocess.py: 
- transform.py: Change the tree structure by specifying one leaf node as root
- gensample.py: Generate training samples from tree structures