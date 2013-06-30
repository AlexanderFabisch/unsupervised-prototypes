import numpy

def logistic_sigmoid(A):
    return 1/(1+numpy.exp(-A))

class RestrictedBoltzmannMachine(object):
    def __init__(self, n_filters, n_epochs, batch_size=1, alpha=0.1, eta=0.0,
                 l1_penalty=0.0, l2_penalty=0.0, cd_n=1, std_dev=0.01, verbose=False):
        self.n_filters = n_filters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.eta = eta
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.cd_n = cd_n
        self.std_dev = std_dev
        self.verbose = verbose

    def __forward(self):
        self.PH = logistic_sigmoid(self.V.dot(self.W.T) + self.bh)
        self.H = self.PH > numpy.random.rand(*self.PH.shape)

    def __backward(self):
        self.PV = logistic_sigmoid(self.H.dot(self.W) + self.bv)
        self.V = self.PV > numpy.random.rand(*self.PV.shape)

    def __reality(self):
        self.__forward()
        return self.PH.T.dot(self.V), self.V.sum(axis=0), self.PH.sum(axis=0)

    def __daydream(self, n):
        for _ in range(n):
            self.__backward()
            self.__forward()
        return self.PH.T.dot(self.PV), self.PV.sum(axis=0), self.PH.sum(axis=0)

    def fit(self, X):
        self.n_samples = X.shape[0]
        self.n_inputs = X.shape[1]

        self.W = numpy.random.randn(self.n_filters, self.n_inputs)
        self.bh = numpy.zeros(self.n_filters)
        self.bv = numpy.zeros(self.n_inputs)

        self.W_mom = numpy.zeros_like(self.W)
        self.bh_mom = numpy.zeros_like(self.bh)
        self.bv_mom = numpy.zeros_like(self.bv)

        for epoch in range(self.n_epochs):
            indices = range(self.n_samples)
            numpy.random.shuffle(indices)
            n_batches = self.n_samples / self.batch_size
            for b in range(n_batches):
                batch = indices[b*self.batch_size:(b+1)*self.batch_size]
                self.V = X[batch]
                pos, pos_bv, pos_bh = self.__reality()
                neg, neg_bv, neg_bh = self.__daydream(self.cd_n)

                self.W_mom = self.alpha / self.batch_size * (pos - neg) + \
                    self.eta * self.W_mom
                self.W += self.W_mom
                if self.l1_penalty > 0:
                    self.W -= self.alpha * self.l1_penalty * numpy.sign(self.W)
                if self.l2_penalty > 0:
                    self.W -= self.alpha * self.l1_penalty * self.W
                # Do not apply the sparsitiy constraint on biases!
                self.bv_mom = self.alpha / self.batch_size * (pos_bv - neg_bv) + \
                    self.eta * self.bv_mom
                self.bv += self.bv_mom
                self.bh_mom = self.alpha / self.batch_size * (pos_bh - neg_bh) + \
                    self.eta * self.bh_mom
                self.bh += self.bh_mom
            if self.verbose:
                print("Finished iteration %d" % (epoch+1))

        self.W_ = self.W

    def predict(self, X):
        self.V = X
        self.__forward()
        self.__backward()
        return self.PV
