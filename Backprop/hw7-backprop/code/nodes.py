"""Computation graph node types

Nodes must implement the following methods:
__init__   - initialize node
forward    - (step 1 of backprop) retrieve output ("out") of predecessor nodes (if
             applicable), update own output ("out"), and set gradient ("d_out") to zero
backward   - (step 2 of backprop), assumes that forward pass has run before.
             Also assumes that backward has been called on all of the node's
             successor nodes, so that self.d_out contains the
             gradient of the graph output with respect to the node output.
             Backward computes summands of the derivative of graph output with
             respect to the inputs of the node, corresponding to paths through the graph
             that go from the node's input through the node to the graph's output.
             These summands are added to the input node's d_out array.
get_predecessors - return a list of the node's parents

Nodes must furthermore have a the following attributes:
node_name  - node's name (a string)
out      - node's output
d_out    - derivative of graph output w.r.t. node output

This computation graph framework was designed and implemented by
Philipp Meerkamp, Pierre Garapon, and David Rosenberg.
License: Creative Commons Attribution 4.0 International License
"""

import numpy as np

class ValueNode(object):
    """Computation graph node having no input but simply holding a value"""
    def __init__(self, node_name):
        self.node_name = node_name
        self.out = None
        self.d_out = None

    def forward(self):
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        pass
        #return self.d_out

    def get_predecessors(self):
        return []

class VectorScalarAffineNode(object):
    """ Node computing an affine function mapping a vector to a scalar."""
    def __init__(self, x, w, b, node_name):
        """ 
        Parameters:
        x: node for which x.out is a 1D numpy array
        w: node for which w.out is a 1D numpy array of same size as x.out
        b: node for which b.out is a numpy scalar (i.e. 0dim array)
        node_name: node's name (a string)
        """
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.x = x
        self.w = w
        self.b = b

    def forward(self):
        self.out = np.dot(self.x.out, self.w.out) + self.b.out
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        d_x = np.dot(self.d_out, self.w.out)
        d_w = np.dot(self.d_out , self.x.out)
        d_b = self.d_out
        self.x.d_out += d_x
        self.w.d_out += d_w
        self.b.d_out += d_b
        return self.d_out

    def get_predecessors(self):
        return [self.x, self.w, self.b]

class SquaredL2DistanceNode(object):

    def __init__(self, a, b, node_name):
        
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.a = a
        self.b = b
        self.a_minus_b = None

    def forward(self):
        self.a_minus_b = self.a.out - self.b.out
        self.out = np.sum(self.a_minus_b ** 2)
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        d_a = self.d_out * 2 * self.a_minus_b
        d_b = -self.d_out * 2 * self.a_minus_b
        self.a.d_out += d_a
        self.b.d_out += d_b
        return self.d_out

    def get_predecessors(self):
        return [self.a, self.b]

class L2NormPenaltyNode(object):

    def __init__(self, l2_reg, w, node_name):

        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.l2_reg = np.array(l2_reg)
        self.w = w

    def forward(self):
        self.out = self.l2_reg*self.w.out.dot(self.w.out)
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        d_w = self.d_out * 2 * self.l2_reg * self.w.out
        self.w.d_out += d_w
        return self.d_out

    def get_predecessors(self):
        return [self.w]

class SumNode(object):

    def __init__(self, a, b, node_name):

        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.a = a
        self.b = b

    def forward(self):
        self.out = np.sum([self.a.out,self.b.out], axis=0) 
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        d_a = self.d_out
        d_b = self.d_out
        self.a.d_out += d_a
        self.b.d_out += d_b
        return self.d_out

    def get_predecessors(self):
        return [self.a, self.b]

class AffineNode(object):

    def __init__(self, W, x, b, node_name):
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.x = x
        self.b = b
        self.W = W

    def forward(self):
        mult_wx = np.dot(self.W.out,self.x.out)
        self.out = mult_wx + self.b.out.T
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        d_x = np.dot(self.W.out.T, self.d_out)
        d_W = np.outer(self.d_out, self.x.out.T)
        d_b = self.d_out
        self.x.d_out += d_x
        self.W.d_out += d_W
        self.b.d_out += d_b
        return self.d_out

    def get_predecessors(self):
        return [self.x, self.W, self.b]


class TanhNode(object):
    def __init__(self, a, node_name):
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.a = a

    def forward(self):
        self.out = np.tanh(self.a.out) 
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        d_a = self.d_out * (np.ones((self.a.out.shape)) - \
                            np.power(np.tanh(self.a.out),2))  
        self.a.d_out += d_a
        return self.d_out

    def get_predecessors(self):
        return [self.a]


class SoftMaxNode(object):
    def __init__(self, a, node_name):
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.a = a
        
    def forward(self):
        self.out = np.exp(self.a.out) / float(sum(np.exp(self.a.out)))
        self.d_out = np.zeros(self.out.shape)
        return self.out
    
    def backward(self):
        self.a.d_out += d_a
        return self.d_out

    def get_predecessors(self):
        return [self.a]


class NegLogLikelihoodNode(object):
    def __init__(self, softmax_X, y, node_name):
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.softmax_X = softmax_X
        self.y = y

    def forward(self):
        m = self.y.out.shape[0]
        p = self.softmax_X.out
        log_likelihood = -np.log(p[range(m),y])
        loss = np.sum(log_likelihood) / m
        self.out = loss
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        d_softmax_X = self.d_out * -1 * np.divide(self.y.out, \
                                                  self.softmax_X.out)  
        self.d_softmax_X.d_out += d_softmax_X
        return self.d_out

    def get_predecessors(self):
        return [self.softmax_X]