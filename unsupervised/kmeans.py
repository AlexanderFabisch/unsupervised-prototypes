import numpy

class KMeans(object):
    def __init__(self, n_filters, batch_size, n_iterations):
        self.n_filters = n_filters
        self.batch_size = batch_size
        self.n_iterations = n_iterations

    def fit(self, X):
        self.X = X
        n_samples = len(X)
        indices = list(range(n_samples))
        numpy.random.shuffle(indices)
        self.C_ = numpy.copy(X[:self.n_filters])
        self.v = numpy.zeros(self.n_filters)
        for i in range(self.n_iterations):
            numpy.random.shuffle(indices)
            for n in range(0, n_samples, self.batch_size):
                M = self.X[n:min(n+self.batch_size, n_samples)]
                d = [numpy.argmin(numpy.sum((self.C_ - M[j])**2, axis=1))
                     for j in range(self.batch_size)]
                for j in range(self.batch_size):
                    c = d[j]
                    self.v[c] += 1
                    eta = 1.0 / self.v[c]
                    self.C_[c] = (1.0-eta) * self.C_[c] + eta * M[j]
            print "Iteration #%d finished." % (i+1)

    def predict(self, X):
        Z = numpy.array([numpy.sum((X[j] - self.C_)**2, axis=1)
                         for j in range(len(X))])
        MU = Z.mean(axis=1)
        return numpy.max((MU[:, numpy.newaxis] - Z, numpy.zeros_like(Z)), axis=0)
