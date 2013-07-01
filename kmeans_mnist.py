import numpy
import pylab
from kmeans import KMeans
from mnist import read
from openann import *


def load(dataset_type, n_samples):
    images, targets = read(range(10), dataset_type)
    images = images.reshape(-1, 784)[:n_samples] / 255.0
    targets = targets[:n_samples].flatten()
    T = numpy.zeros((n_samples, 10))
    T[(range(n_samples), targets)] = 1.0
    return images, T

def scale_features(X, mean, std):
    return (X - mean) / std


if __name__ == "__main__":
    numpy.random.seed(0)

    train_images, T = load("training", 5000)
    test_images, T2 = load("testing", 5000)

    n_filters = 196
    estimator = KMeans(n_filters=n_filters, batch_size=1000, n_iterations=5)
    estimator.fit(train_images)
    X = estimator.predict(train_images)
    X2 = estimator.predict(test_images)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = scale_features(X, X_mean, X_std)
    X2 = scale_features(X2, X_mean, X_std)
    ds = Dataset(X, T)
    vs = Dataset(X2, T2)

    net = Net()
    net.input_layer(ds.inputs())
    net.output_layer(ds.outputs(), Activation.LINEAR)
    net.set_error_function(Error.CE)
    opt = MBSGD({"maximal_iterations" : 100})
    opt.optimize(net, ds)
    print classification_hits(net, ds)
    print numpy.array(confusion_matrix(net, ds), dtype=numpy.int)
    print classification_hits(net, vs)
    print numpy.array(confusion_matrix(net, vs), dtype=numpy.int)

    pylab.figure()
    pylab.subplots_adjust(wspace=0.0, hspace=0.0)
    for i in range(estimator.C_.shape[0]):
        rows = int(numpy.sqrt(n_filters))
        cols = int(numpy.sqrt(n_filters))
        pylab.subplot(rows, cols, i + 1)
        pylab.imshow(estimator.C_[i].reshape(28, 28),
                     cmap=pylab.cm.gray, interpolation="nearest")
        pylab.xticks(())
        pylab.yticks(())
    pylab.show()
