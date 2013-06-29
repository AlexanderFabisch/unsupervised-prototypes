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
            return self.__grad(theta)

        theta = self.W.ravel()
        theta, error, _ = fmin_l_bfgs_b(error_grad, theta, iprint=1,
                                        maxfun=self.maxfun)
        self.W_ = theta.reshape(self.n_filters, self.n_inputs)

    def __grad(self, theta):
        W = theta.reshape(self.n_filters, self.n_inputs)

        # Compute unnormalized features by multiplying weight matrix with data
        F = self.X.dot(W.T) # Linear Activation
        Fs = numpy.sqrt(F ** 2 + 1e-8) # Soft-Absolute Activation

        # Normalize each feature to be equally active by dividing each
        # feature by its l2-norm across all examples
        L2Fs = numpy.apply_along_axis(numpy.linalg.norm, 0, Fs)
        NFs = (Fs / L2Fs[None, :]).T
        # Normalize features per example, so that they lie on the unit l2 -ball
        L2Fn = numpy.apply_along_axis(numpy.linalg.norm, 0, NFs)
        Fhat = NFs.T / L2Fn[:, None]
        # Compute sparsity of each feature over all example, i.e., compute
        # its l1-norm; the objective function is the sum over these
        # sparsities
        error = numpy.apply_along_axis(numpy.linalg.norm, 1, Fhat, 1).sum()
        # Backprop through each feedforward step
        deltaW = self.__l2grad(NFs.T, Fhat, L2Fn, numpy.ones_like(Fhat))
        deltaW = self.__l2grad(Fs.T, NFs, L2Fs, deltaW.T)
        deltaW = (deltaW * (F.T / Fs.T)).dot(self.X)

        return error, deltaW.ravel()

    def __l2grad(self, X, Y, N, D):
        # Backpropagate through normalization
        return D / N[:, None] - Y * (D * X).sum(1)[:, None] / (N ** 2)[:, None]
