# Python2 compability
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy
import sys
import math

numpy.set_printoptions(suppress=True)

class Camera:
    @staticmethod
    def __rq_decomposition(A):
        Q,R = numpy.linalg.qr(numpy.flipud(A).T)
        R = numpy.flipud(R.T)
        Q = Q.T
        return R[:,::-1], Q[::-1,:]

    def __init__(self, file2d, file3d):
        self.txt = numpy.loadtxt(file2d)
        self.txt_len = self.txt.shape[0]
        self.uv = numpy.loadtxt(file3d)
        if self.txt_len != self.uv.shape[0]:
            raise Exception("Bad input!")
        #
        A = numpy.empty((2*self.txt_len, 12))
        for line in range(self.txt_len):
            xyzw = numpy.concatenate([self.txt[line], [1]])
            A[2*line] = numpy.concatenate([xyzw, [0,0,0,0], [ -1*self.uv[line][0]*l for l in xyzw ]])
            A[2*line+1] = numpy.concatenate([[0,0,0,0], xyzw, [ -1*self.uv[line][1]*l for l in xyzw ]])
        #
        U, S, V = numpy.linalg.svd(A)
        self.M = numpy.reshape(V[-1:,], (3,-1))

        tmp_K, tmp_R = self.__rq_decomposition(self.M[:,:3])

        # make diagonal of K positive
        tmp = numpy.diag(numpy.sign(numpy.diag(tmp_K)))
        if numpy.linalg.det(tmp) < 0:
           tmp[1,1] *= -1

        self.K = numpy.dot(tmp_K, tmp)  # camera parameters
        self.R = numpy.dot(tmp, tmp_R)  # camera rotation matrix
        self.T = numpy.linalg.inv(self.K).dot(self.M[:,-1])  # world origin
        self.C = -(self.R.T).dot(self.T) # camera center

    #
    def check_M(self):
        for line in range(self.txt_len):
            xyzw = numpy.concatenate([self.txt[line], [1]])
            new_uv = self.M.dot(numpy.reshape(xyzw, (4,1)))
            xdiff = self.uv[line][0] - new_uv[0]/new_uv[2]
            ydiff = self.uv[line][1] - new_uv[1]/new_uv[2]
            diff = math.sqrt(xdiff**2 + ydiff**2)
            if diff >= 0.01:
                print("Big diff", diff, "for point nr", line, "!")
