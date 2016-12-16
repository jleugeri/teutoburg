import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import teutoburg
from pylab import *
import sympy as sp

num_data_dims = 1
num_label_dims = 1
num_train_data = 500
num_trees= 100
num_lvls = 5
num_test_data = 100
test_scale = 5
num_features = 10
numThresholds = 10

npp = 100
verbose = False

def f(x):
    return x**3-2*x+2

x = np.random.randn(num_train_data)
y = f(x)

training_data = x.reshape((-1,1))
training_labels = y.reshape((-1,1))

testing_data = np.linspace(training_data.min(),training_data.max(),num_test_data).reshape((-1,1))
forest = teutoburg.trainRegressionForest(training_data, training_labels, num_trees, num_features, numThresholds, num_lvls, verbose)
res  = forest(testing_data)
y = array(res).squeeze()

figure()

plot(training_data, training_labels, 'k.')
plot(testing_data, f(testing_data), 'k-')
plot(testing_data, y[:, 0], 'r-')
plot(testing_data, y[:, 0]-y[:, 1], 'r--')
plot(testing_data, y[:, 0]+y[:, 1], 'r--')

show()
