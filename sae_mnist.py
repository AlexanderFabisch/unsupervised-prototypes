import numpy
import pylab
from sae import SparseAutoEncoder
from mnist import read


if __name__ == "__main__":
    numpy.random.seed(0)

    train_images, _ = read(range(10), "training")
    train_images = train_images.reshape(-1, 784)[:10000] / 255.0

    n_filters = 196
    estimator = SparseAutoEncoder(n_filters=n_filters, lmbd=3e-3, beta=3,
                                  sparsity_param=0.1, maxfun=400,
                                  verbose=True)
    estimator.fit(train_images)

    reconstructed = estimator.predict(train_images)
    error = numpy.sum((reconstructed - train_images)**2)
    print("Reconstruction error = %f" % error)

    pylab.figure()
    pylab.subplots_adjust(wspace=0.0, hspace=0.0)
    for i in range(estimator.W1_.shape[0]):
        rows = int(numpy.sqrt(n_filters))
        cols = int(numpy.sqrt(n_filters))
        pylab.subplot(rows, cols, i + 1)
        pylab.imshow(estimator.W1_[i].reshape(28, 28),
                     cmap=pylab.cm.gray, interpolation="nearest")
        pylab.xticks(())
        pylab.yticks(())
    pylab.show()
