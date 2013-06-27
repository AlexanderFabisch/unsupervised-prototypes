import numpy
import pylab
from scipy.optimize import * # TODO include only one optimizer
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.datasets import fetch_olivetti_faces


def check_grad(fun, grad, theta, eps=1e-10):
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
    def __init__(self, n_filters, lmbd, beta, sparsityParam, std_dev):
        self.n_filters = n_filters
        self.lmbd = lmbd
        self.beta = beta
        self.sparsityParam = sparsityParam
        self.std_dev = std_dev

    def fit(self, X):
        self.n_samples = X.shape[0]
        self.n_inputs = X.shape[1]
        self.X = X
        self.indices = (self.n_filters*self.n_inputs,
                        2*self.n_filters*self.n_inputs,
                        2*self.n_filters*self.n_inputs+self.n_filters)
        theta = numpy.random.randn(self.n_filters*self.n_inputs*2
            + self.n_filters + self.n_inputs) * self.std_dev

        def objective(theta):
            return self.error_grad(theta)
        def error(theta):
            return self.error(theta)
        def grad(theta):
            return self.grad(theta)
        grad_error = check_grad(error, grad, theta)
        assert grad_error < 1e-10, "Gradient error = %f" % grad_error

        #theta = fmin(error, theta, maxiter=50, full_output=True, disp=True)
        #theta = fmin_ncg(error, theta, grad)
        theta, s, d = fmin_l_bfgs_b(objective, theta,
                               maxfun=50, iprint=1) # TODO params
        W1, W2, b1, b2 = self.__vector_to_matrices(theta)
        self.W_ = W1
    
    def __vector_to_matrices(self, theta):
        w1, w2, b1, b2 = numpy.split(theta, self.indices)
        W1 = w1.reshape((self.n_filters, self.n_inputs))
        W2 = w2.reshape((self.n_inputs, self.n_filters))
        return W1, W2, b1, b2

    def __forward(self, X):
        A1 = X.dot(self.W1.T) + self.b1
        Z1 = sigmoid(A1)
        A2 = Z1.dot(self.W2.T) + self.b2
        Z2 = sigmoid(A2)
        return Z1, Z2
    
    def __cost(self, meanZ1, Z2, X):
        dEdZ2 = Z2-X
        cost = numpy.sum(dEdZ2**2) / (2*self.n_samples)
        cost += self.lmbd/2 * numpy.sum(self.W1**2) + numpy.sum(self.W2**2)
        cost += self.beta * \
            numpy.sum(self.sparsityParam *
                      numpy.log(self.sparsityParam / meanZ1) +
                      (1 - self.sparsityParam) *
                      numpy.log((1 - self.sparsityParam) / (1 - meanZ1)))
        return cost
    
    def __grad(self, Z1, meanZ1, Z2, X):
        dEdZ2 = Z2-X
        GD2 = sigmoid_der(Z2)
        Delta2 = dEdZ2 * GD2
        W2d = Delta2.T.dot(Z1)/self.n_samples + self.lmbd * self.W2
        b2d = Delta2.mean(axis=0)

        dEdZ1 = Delta2.dot(self.W2)
        GD1 = sigmoid_der(Z1)
        sparse = -self.sparsityParam / meanZ1 + (1-self.sparsityParam) / (1-meanZ1)
        Delta1 = (dEdZ1 + self.beta * sparse) * GD1
        W1d = Delta1.T.dot(X)/self.n_samples + self.lmbd * self.W1
        b1d = Delta1.mean(axis=0)
        assert W1d.shape == self.W1.shape
        assert W2d.shape == self.W2.shape
        assert b1d.shape == self.b1.shape
        assert b2d.shape == self.b2.shape

        grad = numpy.concatenate((W1d.flatten(), W2d.flatten(), b1d, b2d))
        return grad
    
    def error(self, theta):
        self.W1, self.W2, self.b1, self.b2 = self.__vector_to_matrices(theta)
        Z1, Z2 = self.__forward(self.X)
        cost = self.__cost(Z1.mean(axis=0), Z2, self.X)
        return cost
    
    def grad(self, theta):
        self.W1, self.W2, self.b1, self.b2 = self.__vector_to_matrices(theta)
        Z1, Z2 = self.__forward(self.X)
        grad = self.__grad(Z1, Z1.mean(axis=0), Z2, self.X)
        assert grad.shape == theta.shape, str(grad.shape) + "!=" + str(theta.shape)
        return grad
    
    def error_grad(self, theta):
        self.W1, self.W2, self.b1, self.b2 = self.__vector_to_matrices(theta)
        Z1, Z2 = self.__forward(self.X)
        meanZ1 = Z1.mean(axis=0)
        cost = self.__cost(meanZ1, Z2, self.X)
        grad = self.__grad(Z1, meanZ1, Z2, self.X)
        assert grad.shape == theta.shape, str(grad.shape) + "!=" + str(theta.shape)
        return cost, grad

if __name__ == "__main__":
    numpy.random.seed(0)

    patch_width = 2#16
    n_patches = 1#25
    n_filters = 2#64
    maxfun = 50

    dataset = fetch_olivetti_faces(shuffle=True)
    faces = dataset.data

    n_samples, n_features = faces.shape

    # global centering
    faces_centered = faces - faces.mean(axis=0)

    # local centering
    faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)
    faces_centered = faces_centered.reshape(n_samples, 64, 64)

    print("Dataset consists of %d faces" % n_samples)

    patches = [extract_patches_2d(faces_centered[i], (patch_width, patch_width),
                                max_patches=n_patches, random_state=i)
            for i in range(n_samples)]
    patches = numpy.array(patches).reshape(-1, patch_width * patch_width)

    estimator = SparseAutoEncoder(n_filters=n_filters,
                                  lmbd=0.00001,
                                  beta=3,
                                  sparsityParam=0.01,
                                  std_dev=1.0)
    estimator.fit(patches)
    #estimator.fit(patches[numpy.newaxis, 0])

    # Some plotting
    pylab.figure(0)
    for i in range(estimator.W_.shape[0]):
        pylab.subplot(int(numpy.sqrt(n_filters)), int(numpy.sqrt(n_filters)), i + 1)
        pylab.pcolor(estimator.W_[i].reshape(patch_width, patch_width),
                     cmap=pylab.cm.gray)
        pylab.xticks(())
        pylab.yticks(())
    pylab.show()
