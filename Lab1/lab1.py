# Python2 compability
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from camera import Camera
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

c = Camera("./data/task12/pts3d-norm.txt", "./data/task12/pts2d-norm-pic_a.txt")

c.check_M()

print("Projection matrix:\n", c.M, "\n")
print("Camera internals:\n", c.K, "\n")
print("Camera rotation:\n", c.R, "\n")
print("World origin:\n", c.T, "\n")
print("Camera center:\n", c.C, "\n")

plot = pyplot.figure().add_subplot(111, projection='3d')

plot.scatter(*c.C, c='r')

for point in c.txt:
    plot.scatter(*point, c='b')

pyplot.show()
