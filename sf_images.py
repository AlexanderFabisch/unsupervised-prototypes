import numpy
import pylab
from scipy.io import loadmat
from sklearn.feature_extraction.image import extract_patches_2d
from sf import SparseFiltering


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

    estimator = SparseFiltering(n_filters=n_filters, maxfun=200, verbose=True)
    estimator.fit(patches)

    pylab.figure()
    for i in range(estimator.W_.shape[0]):
        rows = int(numpy.sqrt(n_filters))
        cols = int(numpy.sqrt(n_filters))
        pylab.subplot(rows, cols, i + 1)
        pylab.imshow(estimator.W_[i].reshape(patch_width, patch_width),
                     cmap=pylab.cm.gray, interpolation="nearest")
        pylab.xticks(())
        pylab.yticks(())
    pylab.show()
