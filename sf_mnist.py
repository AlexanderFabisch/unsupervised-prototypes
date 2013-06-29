import numpy
import pylab
from sf import SparseFiltering
from mnist import read


if __name__ == "__main__":
    numpy.random.seed(0)

    train_images, _ = read(range(10), "training")
    train_images = train_images.reshape(-1, 784)[:10000] / 255.0

    n_filters = 196
    estimator = SparseFiltering(n_filters=n_filters, maxfun=500,
                                verbose=True)
    estimator.fit(train_images)

    pylab.figure()
    pylab.subplots_adjust(wspace=0.0, hspace=0.0)
    for i in range(estimator.W_.shape[0]):
        rows = int(numpy.sqrt(n_filters))
        cols = int(numpy.sqrt(n_filters))
        pylab.subplot(rows, cols, i + 1)
        pylab.imshow(estimator.W_[i].reshape(28, 28),
                     cmap=pylab.cm.gray, interpolation="nearest")
        pylab.xticks(())
        pylab.yticks(())
    pylab.show()
