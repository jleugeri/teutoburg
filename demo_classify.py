import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import teutoburg
from pylab import *
import sympy as sp
from itertools import repeat

num_clusters = 3
num_data_per_cluster = 100
num_dims = 2
num_trees= 100
num_lvls = 3
num_test_data = 100
test_scale = 5
num_features = 10
numThresholds = 10

npp = 100
verbose = False


def lineFromNormal(lnorm, loff):
    p0 = sp.Point(0, 0)
    p1 = lnorm

    new_dir = sp.Line(p0, p1)
    return new_dir.perpendicular_line(p1*loff)


def intersect2D(new_line, old_dir, old_line):
    ps = new_line.intersection(old_line)

    # Line lies on older line
    if new_line.is_similar(old_line):
        return None

    # No intersections -> parallel line -> no need to cut line
    if len(ps) == 0:
        return new_line

    p0 = ps[0]
    # Some intersection -> cut line
    if isinstance(new_line, sp.Line):
        p = new_line.p2 if new_line.p1.equals(p0) else new_line.p1
        p1 = p if (p-p0).dot(old_dir) > 0 else p0*2 -p
        new_line = sp.Ray(p0, p1)
    elif isinstance(new_line, sp.Ray):
        p = new_line.p1
        if (p-p0).dot(old_dir) > 0:
            new_line = sp.Segment(p0, p)
        else:
            new_line = sp.Ray(p0, p0*2-p)
    elif isinstance(new_line, sp.Segment):
        p = new_line.p1
        p1 = p if (p-p0).dot(old_dir) > 0 else new_line.p2
        new_line = sp.Ray(p0, p1)
    else:
        raise Exception("Must be line or ray or segment!")
    return new_line


def intersectAll2D(new_line, old_dirs_and_lines):
    for (old_dir, old_line) in old_dirs_and_lines:
        new_line = intersect2D(new_line, old_dir, old_line)
    return new_line


def plotLine(line, **kwargs):
    (s, smin, smax) = line.plot_interval()
    pts = [line.arbitrary_point().subs(s, val).n() for val in (smin, smax)]
    return plt.Line2D((pts[0].x, pts[1].x), (pts[0].y, pts[1].y), **kwargs)
"""
def plot2DTree(tree,ax):
    levels = int(np.log2(len(tree)+1))
    plines = dict()
    dirs  = [[]]
    lines = [[]]
    width = np.logspace(1, 0, levels)
    i = 0
    for l in range(levels):
        new_dirs = []
        new_lines = []
        for n in range(2**l):
            if(tree[i].isSplit):
                new_dir = sp.Point(*tree[i].feature)
                new_off = tree[i].threshold
                new_line = intersectAll2D(lineFromNormal(new_dir, new_off), zip(dirs[n], lines[n]))
                pline = plotLine(new_line, linewidth=width[l])
                ax.add_line(pline, )
                plines[(l,n)] = pline
                new_dirs.extend([dirs[n]+[-new_dir], dirs[n]+[new_dir]])
                new_lines.extend([lines[n]+[new_line], lines[n]+[new_line]])
            else:
                new_lines.extend([None, None])
                new_dirs.extend([None, None])
            i += 1
        dirs = new_dirs
        lines = new_lines
    return plines

"""

"""
def getLeafPatches(tree):
    num_nodes = len(tree)
    patches = []
    # Go through all of the leaf nodes
    for i in range(num_nodes-(num_nodes+1)/2, num_nodes):
        lines = [(i-1)//2]
"""



training_data = np.vstack([np.random.randn(num_data_per_cluster, num_dims) + 2*np.random.randn(num_dims) for i in range(num_clusters)])
training_labels = np.array([0]*num_data_per_cluster + [1]*num_data_per_cluster + [2]*num_data_per_cluster)

#test_data = np.random.randn(num_test_data, num_dims)*test_scale
x = np.linspace(training_data[:, 0].min(), training_data[:, 0].max(), npp)
y = np.linspace(training_data[:, 1].min(), training_data[:, 1].max(), npp)
xx,yy = np.meshgrid(x, y)
test_data = np.hstack([xx.reshape((-1, 1)), yy.reshape((-1, 1))])

f = teutoburg.trainClassificationForest(list(zip(training_data, training_labels)), num_trees, num_features, numThresholds, num_lvls, verbose)

res  = f(list(zip(test_data, repeat(None))))

hists = []
for d_resp in res:
    hist = 0
    n = 0
    for l, r in d_resp:
        hist += np.array([r.get(0, 0), r.get(1, 0), r.get(2, 0)], dtype=float)/(r.get(0, 1)+r.get(1, 1)+r.get(2, 1))
        n += 1
    hist /= n;
    hists.append(hist)

#figure()
ax = subplot(111)
#scatter(test_data[:, 0], test_data[:, 1], marker='x', color=res)
imshow(np.array(hists).reshape((npp,npp,3)), extent=(x.min(), x.max(), y.min(), y.max()), origin="lower")

plot(training_data[   :100, 0], training_data[   :100, 1], 'ro')
plot(training_data[100:200, 0], training_data[100:200, 1], 'go')
plot(training_data[200:   , 0], training_data[200:   , 1], 'bo')

xlim([x.min(), x.max()])
ylim([y.min(), y.max()])

#plines = plot2DTree(tree, ax)
show()
