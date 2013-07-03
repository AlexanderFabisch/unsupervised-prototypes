import numpy
import pylab
from unsupervised.kmeans import KMeans

if __name__ == "__main__":
    numpy.random.seed(1)
    X = numpy.vstack((numpy.random.randn(10000, 2)*0.3,
                      numpy.random.randn(10000, 2)*0.3 + numpy.ones(2)))

    estimator = KMeans(2, 200, 10)
    estimator.fit(X)
    print estimator.C_
    print estimator.v
    Y = estimator.predict(X)
    print Y

    pylab.plot(X[:, 0], X[:, 1], "o")
    pylab.plot([estimator.C_[0, 0]], [estimator.C_[0, 1]], "o")
    pylab.plot([estimator.C_[1, 0]], [estimator.C_[1, 1]], "o")
    pylab.show()
