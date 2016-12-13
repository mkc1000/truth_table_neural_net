"""
l0, l1, etc. refer to the nth layer of the neural net. l0 = input layer.
Each l has N rows, where N is the number of data points, and k columns,
where k is the number of features (nodes). Therefore, each row of l0 represents a feature
vector for one data point.

w0, w1 etc. refers to the weights mapping the nth layer to the (n+1)th layer.
If k0 is the number of features (nodes) in l0, and k1 is the number of features in l1,
and l0' is l0 with a column of 1's appended to the left, then w0 has k0+1 rows
and k1 columns, and l1 = sigmoid(l0'*w0). This is for an ordinary neural net layer.

For the truth table mapping, things are more complicated. l1' is l1 with (1-l1) concatenated on the right,
then log'd base e, finally with a column of ones on the left. Four truth
table matrices map to the next layer: tt1a, tt1b, tt1c, and tt1d. For l1
having 4 features, for example tt1a is:
[[a0,a1,a2,a3,a4,a5],
[1,1,1,0,0,0],
[1,0,0,1,1,0],
[0,1,0,1,0,1],
[0,0,1,0,1,1],
[0,0,0,0,0,0],
[0,0,0,0,0,0],
[0,0,0,0,0,0],
[0,0,0,0,0,0]]
All a_i must all be nonpositive.
l2 = exp(l1'*tt1a)+exp(l1'*tt1b)+exp(l1'*tt1c)+exp(l1'*tt1d)
Each node in l2 corresponseds to a columnn of tt1x, and the 1's indicate those nodes in l1
are being combined into that node. l2[0], the first node in l2, for instance, will end up being
e^a0*l1[0]*l1[1]+e^b0*l1[0]*(1-l1[1])+e^c0*(1-l1[0])*l1[1]+e^d0*(1-l1[0])*(1-l1[1])
"""
import numpy as np
import random
from itertools import izip

import cProfile

import matplotlib.pyplot as plt
import seaborn as sns

def sigmoid(arr):
    """
    Takes numpy array, returns numpy array with sigmoid function applied elementwise.
    """
    return 1/(1+np.exp(-arr))

def make_tt(n, coef, for_deriv=False, exponentiated_tt_coefs=True):
    """
    Creates a truth-table-mapping-matrix when there are n nodes in the starting layer. The first row,
    where parameters will be placed, is initialized with random negative numbers.
    Coefficient may be 0,1,2, or 3, and corresponds to the parameter a,b,c, or d.
    """
    m = n*(n-1)/2
    output = np.zeros((2*n+1,m))
    half_1 = coef / 2
    half_2 = coef % 2
    k = 0
    for i in xrange(n):
        for j in xrange(i+1,n):
            output[i+1+n*half_1][k] = 1
            output[j+1+n*half_2][k] = 1
            k+=1
    for i in xrange(m):
        if for_deriv:
            output[0][i] = 1
        else:
            p = random.random()
            if exponentiated_tt_coefs:
                p = np.log(p)
            output[0][i] = p
    return output

def make_tt_back_transformer(n):
    output = np.zeros((2*n+1,n))
    for i in xrange(n):
        output[i+1][i] = 1
        output[i+n+1][i] = -1
    return output

class NeuralNet(object):
    def __init__(self, layer_sizes, net_types,exponentiated_tt_coefs=True):
        """
        Layer_sizes is a list of the number of nodes in each layer.
        net_types specifies wheter it is an ordinary neural net layer 'ord' or truth table 'tt'
        """
        #Validate Inputs
        if len(net_types) != len(layer_sizes) - 1:
            raise ValueError("Layer_sizes must be one element longer than net_types")
        for i, matrix_type in enumerate(net_types):
            if matrix_type != 'ord' and matrix_type != 'tt':
                raise ValueError("net_types must be a list, where each element is 'ord' or 'tt'.")
            if matrix_type == 'tt':
                if layer_sizes[i+1] != layer_sizes[i]*(layer_sizes[i] - 1)/2:
                    raise ValueError("When net_type is 'tt', the layer following it must have (n**2 - n)/2 nodes where n is the number of nodes in the previous layer.")

        #Initalize Variables
        self.layer_sizes = layer_sizes
        self.net_types = net_types
        self.exponentiated_tt_coefs = exponentiated_tt_coefs
        self.n_layers = len(self.layer_sizes)
        self.layers = [None for _ in layer_sizes]
        self.tt_layers_components = [None for _ in layer_sizes]
        self.dcostdlayers = [None for _ in layer_sizes]
        self.nets = []
        self.dcostdnets = [None for _ in xrange(self.n_layers - 1)]
        for i in xrange(self.n_layers - 1):
            if net_types[i] == 'ord':
                mat_rows = self.layer_sizes[i]+1
                mat_cols = self.layer_sizes[i+1]
                self.nets.append(np.random.rand(mat_rows,mat_cols)-0.5)
                #
                #
                #
                # TODO: come back to initializing randomly in a way that makes sense
                # with the P(e|h) stuff
                #
                #
                #
            else: #net_types[i] == 'tt'
                n = layer_sizes[i]
                lst_of_tts = [make_tt(n, coef, exponentiated_tt_coefs=self.exponentiated_tt_coefs) for coef in xrange(4)]
                self.nets.append(lst_of_tts)
        self.ground_truth = None
        self.cost = lambda: 0.5*np.mean((self.layers[-1] - self.ground_truth)**2)
        self.d_cost_func = lambda: self.layers[-1] - self.ground_truth
        self.alpha = 0.1

    def input_data(self,data):
        data = np.array(data)
        if np.size(data, axis=1) != self.layer_sizes[0]:
            raise ValueError('Data be a 2D array with same number of columns as nodes in layer 0.')
        self.layers[0] = data

    def input_ground_truth(self,truth):
        truth = np.array(truth)
        rows = np.shape(truth)[0]
        truth = truth.reshape(rows,-1)
        if np.shape(truth)[1] != self.layer_sizes[-1]:
            raise ValueError('Ground truth must be a 2D array with same number of columns as nodes in last layer.')
        self.ground_truth = truth

    def feed_forward(self):
        for layer, net in enumerate(self.nets):
            if self.net_types[layer] == 'ord':
                init_layer = np.insert(self.layers[layer], 0, 1, axis=1)
                self.layers[layer+1] = sigmoid(init_layer.dot(net))
            else: #net type is 'tt'
                if not self.exponentiated_tt_coefs:
                    for tt in net:
                        tt[0] = np.log(tt[0])
                tta, ttb, ttc, ttd = net
                init_layer = np.hstack((self.layers[layer], 1-self.layers[layer]))
                init_layer = np.log(init_layer)
                init_layer = np.insert(init_layer, 0, 1, axis=1)
                self.tt_layers_components[layer+1] = [np.exp(np.dot(init_layer,tta)), np.exp(np.dot(init_layer,ttb)), np.exp(np.dot(init_layer,ttc)), np.exp(np.dot(init_layer,ttd))]
                self.layers[layer+1] = sum(self.tt_layers_components[layer+1])
                if not self.exponentiated_tt_coefs:
                    for tt in net:
                        tt[0] = np.exp(tt[0])


    def set_cost_func_and_deriv(self, cost_func, d_cost_func):
        """
        cost_func must be a function take two arguments: ground_truth and predictions, both np arrays of equal shape: (n_data_points x n_features)
        d_cost_func must a function that takes ground_truth and predictions, and returns the partial derivative of cost_func with respect to predictions evaluated at the given prediction.
        """
        self.cost = lambda: cost_func(self.ground_truth, self.layers[-1])
        self.d_cost_func = lambda: d_cost_func(self.ground_truth, self.layers[-1])

    def feed_backward(self):
        self.dcostdlayers[-1] = self.d_cost_func()
        for l, dcostdnet in reversed(list(enumerate(self.dcostdnets))):
            if self.net_types[l] == 'ord':
                init_layer = np.insert(self.layers[l], 0, 1, axis=1)
                data_points = self.layers[l+1].shape[0]
                transformed_output_layer = self.layers[l+1]*(1-self.layers[l+1]) * self.dcostdlayers[l+1]
                self.dcostdnets[l] = init_layer.T.dot(transformed_output_layer)/data_points
                if l > 0:
                    self.dcostdlayers[l] = transformed_output_layer.dot(self.nets[l].T)[:,1:] # The [1:] gets rid of the bias node.
            else: # net type is 'tt'
                self.dcostdnets[l] = [None, None, None, None]
                for i, tt_layer_component in enumerate(self.tt_layers_components[l+1]):
                    if self.exponentiated_tt_coefs:
                        self.dcostdnets[l][i] = np.sum((self.dcostdlayers[l+1] * tt_layer_component), axis=0)
                    else:
                        self.dcostdnets[l][i] = np.sum((self.dcostdlayers[l+1] * tt_layer_component), axis=0) / self.nets[l][i][0]
                if l > 0:
                    back_prop_helper = make_tt_back_transformer(self.layer_sizes[l])
                    self.dcostdlayers[l] = sum([(self.dcostdlayers[l+1]*self.tt_layers_components[l+1][x]).dot(self.nets[l][x].T).dot(back_prop_helper) for x in xrange(4)])*1./self.layers[l]
                    if not self.exponentiated_tt_coefs:
                        raise NotImplementedError("You forgot to implement me, Michael!")
                        #
                        # TODO correct this part for non-log'd-ness of coefficients
                        #

    def set_learning_rate(alpha):
        self.alpha = alpha

    def update_weights(self):
        for i, (net, dcostdnet, net_type) in enumerate(izip(self.nets, self.dcostdnets, self.net_types)):
            if net_type == 'ord':
                self.nets[i] = net - self.alpha*dcostdnet
            else: #net_type == 'tt'
                for x in xrange(4):
                    self.nets[i][x][0] = self.nets[i][x][0] - self.alpha*dcostdnet[x] #The first row of the truth table matrix gets updated
                    for j, weight in enumerate(self.nets[i][x][0]):
                        if self.exponentiated_tt_coefs:
                            if weight > 0:
                                self.nets[i][x][0][j] = 0
                        else:
                            if weight < 0.0001:
                                self.nets[i][x][0][j] = 0.0001
                            if weight > 1:
                                self.nets[i][x][0][j] = 1.

    def train(self, n_iter=1):
        for _ in xrange(n_iter):
            self.feed_forward()
            self.feed_backward()
            self.update_weights()

    def fit(self, X, Y, n_iter=10000):
        self.input_data(X)
        self.input_ground_truth(Y)
        self.train(n_iter)

def plot_progression_weights(nn,lst_indices,n_iter=1000,num_subplots=1):
    """
    nn: the neural net to track
    lst_indices: a list of 3-tuples or 4-tuples of the form (layer,i,j) or (layer,tt_coef,0,j), which accesses nn.nets[layer][i][j] or nn.nets[layer][tt_coef][0][j]
    n_iter: number of training cycles
    """
    def get_weight_from_indices_tuple(arr, indices_tuples):
        loc = arr
        for index in indices_tuples:
            loc = loc[index]
        return loc
    weight_histories = [[] for _ in lst_indices]
    for _ in xrange(n_iter):
        nn.train()
        for i, weight_history in enumerate(weight_histories):
            weight_history.append(get_weight_from_indices_tuple(nn.nets,lst_indices[i]))
    fig = plt.figure()
    num_per_plot = len(lst_indices)/num_subplots
    for i, (weight_history, indices_tuple) in enumerate(zip(weight_histories,lst_indices)):
        ax_at = i/num_per_plot + 1
        ax = fig.add_subplot(1,num_subplots,ax_at)
        #ax.set_ylim([0,1])
        ax.plot(weight_history, label=indices_tuple)
    plt.legend(loc='best')

def make_list_of_weights_to_watch():
    output = []
    for node in xrange(6):
        for coef in xrange(4):
            output.append((0,coef,0,node))
    return output

def main():
    nn = NeuralNet([4,6,1],['tt','ord'])
    input_data = np.random.rand(500,4)
    truth = np.array([row[0]*row[1]>row[2]*row[3] for row in input_data]).reshape(-1,1)
    nn.fit(input_data,truth,n_iter=1)
    print np.sum([np.sum(arr) for arr in nn.dcostdnets]), nn.cost()
    nn.train(30000)
    print np.sum([np.sum(arr) for arr in nn.dcostdnets]), nn.cost()

if __name__ == '__main__':
    #cProfile.run('main()')
    pass
