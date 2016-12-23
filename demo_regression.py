import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import teutoburg
from pylab import *
import sympy as sp
from itertools import repeat

num_data_dims = 1
num_label_dims = 1
num_train_data = 1000
num_trees= 50
num_lvls = 2
num_test_data = 100
test_scale = 5
num_features = 5
numThresholds = 5

npp = 100
verbose = False

def f(x):
    #return (-2*x) + (1)*(x>-0.5) + (1 + 2*x)*(x>0.5)
    return x**3-2*x+2

x = np.hstack((np.random.rand(num_train_data/4)-1.5, np.random.rand(num_train_data/2)-0.5, np.random.rand(num_train_data/4)+0.5))
y = f(x)
y += np.random.randn(*y.shape)*(x+2.0)/3.0

training_data = x.reshape((-1,1))
training_labels = y.reshape((-1,1))

testing_data = np.linspace(training_data.min(),training_data.max(), num_test_data).reshape((-1,1))
forest = teutoburg.trainRegressionForest(list(zip(training_data, training_labels)), num_trees, num_features, numThresholds, num_lvls, verbose)
res  = forest(list(zip(testing_data,repeat(None))))

pred = np.zeros((num_test_data, num_label_dims), dtype=float)
cov  = np.zeros((num_test_data, num_label_dims, num_label_dims), dtype=float)
for i, datapoint_resp in enumerate(res):
    for (d_pred, d_cov) in datapoint_resp:
        pred[i, :] += d_pred
        cov[i, :, :] += d_cov
pred /= num_trees;
cov  /= num_trees;


figure()

plot(training_data, training_labels, 'k.')
plot(testing_data, f(testing_data), 'k-')
plot(testing_data, pred, 'r-')
fill_between(testing_data.squeeze(), pred.squeeze()+np.sqrt(cov.squeeze()), pred.squeeze()-np.sqrt(cov.squeeze()), facecolor=(1.0,0.0,0.0,0.5))

show()
