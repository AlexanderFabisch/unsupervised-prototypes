import numpy
import pylab
from scipy.io import loadmat
from sklearn.feature_extraction.image import extract_patches_2d
from unsupervised.sae import SparseAutoEncoder


if __name__ == "__main__":
    numpy.random.seed(0)

    # Dataset is taken from Stanfords course:
    # http://www.stanford.edu/class/cs294a/sparseAutoencoder.pdf
    # http://ufldl.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder
    images = loadmat("IMAGES")["IMAGES"].T
    def normalize_data(data):
        data = data - numpy.mean(data)
        pstd = 3 * numpy.std(data)
        data = numpy.fmax(numpy.fmin(data, pstd), -pstd) / pstd
        data = (data + 1) * 0.4 + 0.1;
        return data
    images = normalize_data(images)

    patch_width = 8
    n_filters = 25

    n_samples, n_rows, n_cols = images.shape
    n_features = n_rows * n_cols
    patches = [extract_patches_2d(images[i], (patch_width, patch_width),
                                  max_patches=1000, random_state=i)
            for i in range(n_samples)]
    patches = numpy.array(patches).reshape(-1, patch_width * patch_width)
    print("Dataset consists of %d samples" % n_samples)

    estimator = SparseAutoEncoder(n_filters=n_filters,
                                  lmbd=0.0001, beta=3, sparsity_param=0.01,
                                  maxfun=500, verbose=True)
    estimator.fit(patches)

    pylab.figure(1)
    for i in range(estimator.W1_.shape[0]):
        rows = max(int(numpy.sqrt(n_filters)), 2)
        cols = max(int(numpy.sqrt(n_filters)), 2)
        pylab.subplot(rows, cols, i + 1)
        pylab.imshow(estimator.W1_[i].reshape(patch_width, patch_width),
                     cmap=pylab.cm.gray, interpolation="nearest")
        pylab.xticks(())
        pylab.yticks(())

    pylab.figure(2)
    for i in range(estimator.W2_.shape[1]):
        rows = max(int(numpy.sqrt(n_filters)), 2)
        cols = max(int(numpy.sqrt(n_filters)), 2)
        pylab.subplot(rows, cols, i + 1)
        pylab.imshow(estimator.W2_.T[i].reshape(patch_width, patch_width),
                     cmap=pylab.cm.gray, interpolation="nearest")
        pylab.xticks(())
        pylab.yticks(())

    pylab.figure(3)
    n_plot_patches = numpy.min((len(patches), 196))
    for i in range(n_plot_patches):
        dim = int(numpy.sqrt(n_plot_patches))
        pylab.subplot(dim, dim, i + 1)
        pylab.imshow(patches[i].reshape((patch_width, patch_width)),
                     cmap=pylab.cm.gray, interpolation="nearest")
        pylab.xticks(())
        pylab.yticks(())
    pylab.show()
