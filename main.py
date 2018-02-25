import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from functions import *

# init
max_iterations = 8000
n_datasets = 4
n_datapoints = 200
mutation_factor = 1
h = 0.01 # grid view resolution

best_weights = None
for i in range(n_datasets):
    print("Dataset " + str(i+1) + ":")
    #generate the data set
    X, y = sklearn.datasets.make_moons(n_datapoints, noise=0.08)
    y = y*2 -1
    xmin = min(X[:,0])
    xmax = max(X[:,0])
    ymin = min(X[:,1])
    ymax = max(X[:,1])
    #train
    best_weights = train(X,y,max_iterations,mutation_factor,best_weights)

# final evaluation
nn = NN(2,3,1)
# load the best found NN and activate it for plotting
nn.set_weights(best_weights)
# activate all points
results = []
for i in range(len(X)):
    result = nn.activate(X[i])
    results.append(result)
# plot all points as called by the NN, divide per call made (pos/neg)
pos = []
neg = []
for i in range(len(results)):
    if results[i] > 0:
        pos.append(X[i])
    if results[i] < 0:
        neg.append(X[i])
pos = np.array(pos)
neg = np.array(neg)
# calculate errors
error = 0
wrongpoints = []
# iterate all points
for i in range(len(X)):
    correct_result = y[i]
    called_result = call(results[i])
    if called_result != correct_result:
        error += 1
        wrongpoints.append(X[i])
wrongpoints = np.array(wrongpoints)

# plot the initial data
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.title("initial data")
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

# plot the calls made by the NN
plt.subplot(132)
plt.title("datapoint calls")
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.scatter(pos[:,0], pos[:,1], s=40, c='b', cmap=plt.cm.Spectral)
plt.scatter(neg[:,0], neg[:,1], s=40, c='r', cmap=plt.cm.Spectral)

# generate 2 2d grids for the x & y bounds
gridy, gridx = np.mgrid[slice(ymin, ymax + h, h),
                slice(xmin, xmax + h, h)]

z = []
for i in range(len(gridy[:,0])):
    line = []
    for j in range(len(gridx[0])):
        testpoint = [gridx[0][j],gridy[:,0][i]]
        testresult = call(nn.activate(testpoint))
        line.append(testresult)
    z.append(line)
z = np.array(z)

# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
z_min, z_max = -np.abs(z).max(), np.abs(z).max()

plt.subplot(133)
plt.pcolor(gridx, gridy, z, cmap='RdBu', vmin=z_min, vmax=z_max)
plt.title("errors: " + str(error) + " of " + str(len(X)))
# set the limits of the plot to the limits of the data
plt.axis([gridx.min(), gridx.max(), gridy.min(), gridy.max()])

# plot all points, highlight wrong ones
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.scatter(wrongpoints[:,0], wrongpoints[:,1], s=40, c='y', cmap=plt.cm.Spectral)

plt.show()
