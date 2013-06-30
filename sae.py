import numpy
import pylab
from scipy.io import loadmat
from scipy.optimize import fmin_l_bfgs_b
from sklearn.feature_extraction.image import extract_patches_2d


def check_grad(fun, grad, theta, eps=1e-4):
    exact = grad(theta)
    approx = numpy.ndarray(*theta.shape)
    for i in range(len(theta)):
        theta[i] += eps
        fun_p = fun(theta)
        theta[i] -= 2*eps
        fun_m = fun(theta)
        theta -= eps
        approx[i] = (fun_p - fun_m) / (2 * eps)
    print "Compare derivatives wrt %d parameters" % len(theta)
    print "    Exact            Approx."
    print numpy.hstack((exact[:, numpy.newaxis], approx[:, numpy.newaxis]))
    return numpy.sum((exact - approx)**2)


def sigmoid(a):
    return 1 / (1 + numpy.exp(-a))

def sigmoid_der(z):
    return z * (1-z)


class SparseAutoEncoder(object):
    def __init__(self, n_filters, lmbd, beta, sparsity_param, maxfun,
                 verbose=False):
        self.n_filters = n_filters
        self.lmbd = lmbd
        self.beta = beta
        self.sparsity_param = sparsity_param
        self.maxfun = maxfun
        self.verbose = verbose

    def fit(self, X):
        self.n_samples = X.shape[0]
        self.n_inputs = X.shape[1]
        self.X = X
        self.indices = (self.n_filters*self.n_inputs,
                        2*self.n_filters*self.n_inputs,
                        2*self.n_filters*self.n_inputs+self.n_filters)

        r = numpy.sqrt(6) / numpy.sqrt(self.n_filters + self.n_inputs + 1)
        self.W1 = numpy.random.random((self.n_filters, self.n_inputs)) * 2 * r - r
        self.W2 = numpy.random.random((self.n_inputs, self.n_filters)) * 2 * r - r
        self.b1 = numpy.zeros(self.n_filters)
        self.b2 = numpy.zeros(self.n_inputs)

        self.A1 = numpy.ndarray((self.n_samples, self.n_filters))
        self.Z1 = numpy.ndarray((self.n_samples, self.n_filters))
        self.A2 = numpy.ndarray((self.n_samples, self.n_inputs))
        self.Z2 = numpy.ndarray((self.n_samples, self.n_inputs))
        self.GD2 = numpy.ndarray((self.n_samples, self.n_inputs))
        self.dEdZ2 = numpy.ndarray((self.n_samples, self.n_inputs))
        self.Delta2 = numpy.ndarray((self.n_samples, self.n_inputs))
        self.GD1 = numpy.ndarray((self.n_samples, self.n_filters))
        self.dEdZ1 = numpy.ndarray((self.n_samples, self.n_filters))
        self.Delta1 = numpy.ndarray((self.n_samples, self.n_filters))

        theta = self.__fold(self.W1, self.W2, self.b1, self.b2)
        def f(theta):
            return self.error_grad(theta)
        theta, _, _ = fmin_l_bfgs_b(f, theta, maxfun=self.maxfun,
                                    iprint=1 if self.verbose else -1, m=20)

        self.__unfold(theta)
        self.W1_ = self.W1
        self.W2_ = self.W2
        self.b1_ = self.b1
        self.b2_ = self.b2

    def __forward(self):
        self.A1 = self.X.dot(self.W1.T) + self.b1
        self.Z1 = sigmoid(self.A1)
        self.A2 = self.Z1.dot(self.W2.T) + self.b2
        self.Z2 = sigmoid(self.A2)

    def error_grad(self, theta):
        self.__unfold(theta)
        self.__forward()
        self.dEdZ2 = self.Z2 - self.X
        meanZ1 = self.Z1.mean(axis=0)
        return self.__error(meanZ1), self.__grad(meanZ1)

    def __error(self, meanZ1):
        error = numpy.sum(self.dEdZ2**2) / (2*self.n_samples) + \
            self.lmbd/2 * (numpy.sum(self.W1**2) + numpy.sum(self.W2**2)) + \
            self.beta * numpy.sum(self.sparsity_param *
                numpy.log(self.sparsity_param / meanZ1) +
                (1 - self.sparsity_param) *
                numpy.log((1 - self.sparsity_param) / (1 - meanZ1)))
        return error

    def __grad(self, meanZ1):
        self.GD2 = sigmoid_der(self.Z2)
        self.Delta2 = self.dEdZ2 * self.GD2
        W2d = self.Delta2.T.dot(self.Z1)/self.n_samples + self.lmbd * self.W2
        b2d = self.Delta2.mean(axis=0)

        self.dEdZ1 = self.Delta2.dot(self.W2)
        self.GD1 = sigmoid_der(self.Z1)
        sparse = -self.sparsity_param / meanZ1 + (1-self.sparsity_param) / (1-meanZ1)
        self.Delta1 = (self.dEdZ1 + self.beta * sparse) * self.GD1
        W1d = self.Delta1.T.dot(self.X)/self.n_samples + self.lmbd * self.W1
        b1d = self.Delta1.mean(axis=0)

        grad = self.__fold(W1d, W2d, b1d, b2d)
        return grad

    def __fold(self, W1, W2, b1, b2):
        return numpy.concatenate((W1.ravel(), W2.ravel(), b1, b2))

    def __unfold(self, theta):
        W1, W2, self.b1, self.b2 = numpy.split(theta, self.indices)
        self.W1 = W1.reshape(self.n_filters, self.n_inputs)
        self.W2 = W2.reshape(self.n_inputs, self.n_filters)

    def predict(self, X):
        self.X = X
        self.__forward()
        return self.Z2
