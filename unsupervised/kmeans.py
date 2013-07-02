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
		for i in range(self.n_iterations):
			numpy.random.shuffle(indices)
			for n in range(0, n_samples, self.batch_size):
				M = self.X[n:min(n+self.batch_size, n_samples)]
				d = [numpy.argmin(numpy.sum((self.C_ - M[j])**2, axis=1))
				     for j in range(self.batch_size)]
				v = numpy.bincount(d, minlength=self.n_filters)
				eta = 1.0 / numpy.max((v, numpy.ones_like(v)*1e-8), axis=0)
				for j in range(self.batch_size):
					c = d[j]
					self.C_[c] = (1.0-eta[c]) * self.C_[c] + eta[c] * M[j]
			print "Iteration #%d finished." % (i+1)

	def predict(self, X):
		Z = numpy.array([numpy.sum((self.C_ - self.X[j])**2, axis=1)
				    	 for j in range(len(X))])
		MU = Z.mean(axis=1)
		return numpy.max((Z - MU[:, numpy.newaxis], numpy.zeros_like(Z)), axis=0)
