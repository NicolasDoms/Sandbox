import numpy as np
import matplotlib.pyplot as plt
import copy


class Node:
    def __init__(self, size):
        self.output = 0
        self.size = size
        self.inputs = list()
        self.weights = list()
        for i in range(size):
            self.weights.append(np.random.randn())
            self.inputs.append(0)
        self.weights.append(np.random.randn())

    def calculate(self):
        # output = tanh(W1*input1 + W2*input2 + ...)
        intermediate = 0
        for i in range(len(self.inputs)):
            temp = self.inputs[i]*self.weights[i]
            intermediate += temp
        self.output = np.tanh(intermediate + self.weights[i+1])

    def load_inputs(self, inputs):
        if len(inputs) != self.size:
            print("error: wrong size of inputs " + str(len(inputs)) + " for this node of size " + str(self.size))
            self.inputs = [0]*self.size
        else:
            self.inputs = inputs

    def get_output(self):
        return self.output

    def activate(self,inputs):
        self.load_inputs(inputs)
        self.calculate()
        return self.get_output()

    def get_weights(self):
        returnvalue = copy.deepcopy(self.weights)
        return returnvalue

    def set_weights(self,weights):
        self.weights = copy.deepcopy(weights)

    def mutate(self,s):
        for i in range(len(self.weights)):
            r = np.random.rand()
            if r < 0.95:
                self.weights[i] = self.weights[i] + np.random.normal(0,s)
            else:
                # wild mutation!!!
                self.weights[i] = self.weights[i] + np.random.normal(0,s*10)


class NN:
    def __init__(self,inputsize,hiddensize,outputsize):
        self.node1 = Node(1)
        self.node2 = Node(1)

        self.node3 = Node(2)
        self.node4 = Node(2)
        self.node5 = Node(2)
        self.node6 = Node(2)

        self.node7 = Node(4)
        self.node8 = Node(4)
        self.node9 = Node(4)
        self.node10 = Node(4)

        self.node11 = Node(4)
        self.inputsize = inputsize

    def load_inputs(self, inputs):
        if len(inputs) != self.inputsize:
            print("error: wrong size of inputs " + str(len(inputs)) + " for this NN of size " + str(self.inputsize))
            self.inputs = [0]*self.inputsize
        else:
            self.inputs = inputs

    def calculate(self):
        out1 = self.node1.activate([self.inputs[0]])
        out2 = self.node2.activate([self.inputs[1]])

        out3 = self.node3.activate([out1,out2])
        out4 = self.node4.activate([out1,out2])
        out5 = self.node5.activate([out1,out2])
        out6 = self.node6.activate([out1,out2])

        out7 = self.node7.activate([out3,out4,out5,out6])
        out8 = self.node8.activate([out3,out4,out5,out6])
        out9 = self.node9.activate([out3,out4,out5,out6])
        out10 = self.node10.activate([out3,out4,out5,out6])

        out11 = self.node11.activate([out7,out8,out9,out10])

        return out11

    def activate(self,inputs):
        self.load_inputs(inputs)
        result = self.calculate()
        return result

    def get_weights(self):
        w = list()
        w.append(self.node1.get_weights())
        w.append(self.node2.get_weights())
        w.append(self.node3.get_weights())
        w.append(self.node4.get_weights())
        w.append(self.node5.get_weights())
        w.append(self.node6.get_weights())
        w.append(self.node7.get_weights())
        w.append(self.node8.get_weights())
        w.append(self.node9.get_weights())
        w.append(self.node10.get_weights())
        w.append(self.node11.get_weights())
        return w

    def set_weights(self, weights):
        self.node1.set_weights(weights[0])
        self.node2.set_weights(weights[1])
        self.node3.set_weights(weights[2])
        self.node4.set_weights(weights[3])
        self.node5.set_weights(weights[4])
        self.node6.set_weights(weights[5])
        self.node7.set_weights(weights[6])
        self.node8.set_weights(weights[7])
        self.node9.set_weights(weights[8])
        self.node10.set_weights(weights[9])
        self.node11.set_weights(weights[10])

    def mutate(self,s):
        self.node1.mutate(s)
        self.node2.mutate(s)
        self.node3.mutate(s)
        self.node4.mutate(s)
        self.node5.mutate(s)
        self.node6.mutate(s)
        self.node7.mutate(s)
        self.node8.mutate(s)
        self.node9.mutate(s)
        self.node10.mutate(s)
        self.node11.mutate(s)


def call(fl):
    if fl > 0:
        return 1
    else:
        return -1


def train(X, y, max_iterations, mutation_factor, weights=None):
    min_error = len(X)
    nn = NN(2, 3, 1)
    if weights is not None:
        nn.set_weights(weights)
    best_weights = nn.get_weights()
    for iteration in range(max_iterations):
        # activate all points
        results = []
        for i in range(len(X)):
            result = nn.activate(X[i])
            results.append(result)

        # calculate error
        error = 0
        # iterate all points
        for i in range(len(X)):
            correct_result = y[i]
            called_result = call(results[i])
            if called_result != correct_result:
                error += 1
        relative_error = error / len(X)  # between 0 and 1
        s = mutation_factor * relative_error  # will be the sigma to mutate with

        # evaluate
        print("iteration " + str(iteration) + ": [error: " + str(error) + ", best error: " + str(min_error) + "]")
        if error <= min_error:
            print("found new or equal maximum")
            best_weights = nn.get_weights()
            min_error = error
        nn.set_weights(best_weights)
        nn.mutate(s)

        if error < len(X)*0.05:
            break
    return best_weights

def merge(w1,w2):
    for i in range(len(w1)):
        if isinstance(w1[i], list):
            for j in range(len(w1[i])):
                r = np.random.random_integers(0,1)
                if r>0:
                    w1[i][j] = w2[i][j]
        else:
            r = np.random.random_integers(0, 1)
            if r > 0:
                w1[i] = w2[i]
    return w1