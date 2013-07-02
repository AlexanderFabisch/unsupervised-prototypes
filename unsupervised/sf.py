import numpy
from scipy.optimize import fmin_l_bfgs_b

class SparseFiltering(object):
    def __init__(self, n_filters, maxfun, verbose=False):
        self.n_filters = n_filters
        self.maxfun = maxfun
        self.verbose = verbose

    def fit(self, X):
        # Implementation taken from
        # https://github.com/jmetzen/scikit-learn/blob/master/sklearn/decomposition/sparse_filtering.py
        self.n_samples, self.n_inputs = X.shape
        self.X = X

        self.W = numpy.random.random((self.n_filters, self.n_inputs)) * 2 - 1

        def error_grad(theta):
            return self.__error_grad(theta)

        theta = self.W.ravel()
        theta, error, _ = fmin_l_bfgs_b(error_grad, theta, iprint=1,
                                        maxfun=self.maxfun)
        self.W_ = theta.reshape(self.n_filters, self.n_inputs)

    def __error_grad(self, theta):
        self.W = theta.reshape(self.n_filters, self.n_inputs)

        self.__forward()

        # Compute sparsity of each feature over all example, i.e., compute
        # its l1-norm; the objective function is the sum over these
        # sparsities
        error = numpy.apply_along_axis(numpy.linalg.norm, 1, self.Fhat, 1).sum()
        # Backprop through each feedforward step
        Wd = self.__l2grad(self.NFs, self.Fhat, self.L2Fn, numpy.ones_like(self.Fhat))
        Wd = self.__l2grad(self.Fs.T, self.NFs.T, self.L2Fs, Wd.T)
        Wd = (Wd * (self.F / self.Fs).T).dot(self.X)

        return error, Wd.ravel()

    def __l2grad(self, X, Y, N, D):
        # Backpropagate through normalization
        return D / N[:, numpy.newaxis] - \
            Y * (D * X).sum(axis=1)[:, numpy.newaxis] / (N ** 2)[:, numpy.newaxis]

    def __forward(self):
        # Compute unnormalized features by multiplying weight matrix with data
        self.F = self.X.dot(self.W.T) # Linear Activation
        self.Fs = numpy.sqrt(self.F ** 2 + 1e-8) # Soft-Absolute Activation

        # Normalize each feature to be equally active by dividing each
        # feature by its l2-norm across all examples
        self.L2Fs = numpy.apply_along_axis(numpy.linalg.norm, 0, self.Fs)
        self.NFs = self.Fs / self.L2Fs
        # Normalize features per example, so that they lie on the unit l2 -ball
        self.L2Fn = numpy.apply_along_axis(numpy.linalg.norm, 1, self.NFs)
        self.Fhat = self.NFs / self.L2Fn[:, numpy.newaxis]

    def predict(self, X):
        self.X = X
        self.__forward()
