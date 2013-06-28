import numpy
import pylab
from sae import SparseAutoEncoder
from mnist import read


if __name__ == "__main__":
    numpy.random.seed(0)

    train_images, _ = read(range(10), "training")
    train_images = train_images.reshape(-1, 784)[:10000]
    train_images /= 255.0

    n_filters = 196
    estimator = SparseAutoEncoder(n_filters=n_filters, lmbd=3e-3, beta=3,
                                  sparsityParam=0.1, maxfun=400,
                                  verbose=True)
    estimator.fit(train_images)

    pylab.figure(1)
    for i in range(estimator.W1_.shape[0]):
        rows = max(int(numpy.sqrt(n_filters)), 2)
        cols = max(int(numpy.sqrt(n_filters)), 2)
        pylab.subplot(rows, cols, i + 1)
        pylab.imshow(estimator.W1_[i].reshape(28, 28),
                     cmap=pylab.cm.gray, interpolation="nearest")
        pylab.xticks(())
        pylab.yticks(())
    pylab.show()
