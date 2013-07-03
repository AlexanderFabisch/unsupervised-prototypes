import numpy
from mnist import read
from openann import *


def load_mnist(dataset_type, n_samples):
    images, raw_targets = read(range(10), dataset_type)
    images = images.reshape(-1, 784)[:n_samples] / 255.0
    raw_targets = raw_targets[:n_samples].flatten()
    targets = numpy.zeros((n_samples, 10))
    targets[(range(n_samples), raw_targets)] = 1.0
    return images, targets


def scale_features(X, mean, std):
    return (X - mean) / std


def test_classifier(X_train, T_train, X_valid, T_valid):
    ts = Dataset(X_train, T_train)
    vs = Dataset(X_valid, T_valid)

    net = Net()
    net.input_layer(ts.inputs())
    net.output_layer(ts.outputs(), Activation.LINEAR)
    net.set_error_function(Error.CE)
    opt = MBSGD({"maximal_iterations" : 100})
    opt.optimize(net, ts)
    print "Training set:"
    print classification_hits(net, ts)
    print numpy.array(confusion_matrix(net, ts), dtype=numpy.int)
    print "Validation set:"
    print classification_hits(net, vs)
    print numpy.array(confusion_matrix(net, vs), dtype=numpy.int)


def sigmoid(a):
    return 1 / (1 + numpy.exp(-a))

def sigmoid_der(z):
    return z * (1-z)


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
