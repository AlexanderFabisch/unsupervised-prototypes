import numpy
import pylab
from unsupervised.kmeans import KMeans
from tools import load_mnist, scale_features, test_classifier


if __name__ == "__main__":
    numpy.random.seed(0)

    train_images, T = load_mnist("training", 60000)
    test_images, T2 = load_mnist("testing", 10000)
    print "Dataset loaded"

    train_cluster = train_images[:10000]
    train_classifier = train_images
    label_classifier = T
    n_filters = 196
    estimator = KMeans(n_filters=n_filters, batch_size=1000, n_iterations=10)
    estimator.fit(train_cluster)
    X = estimator.predict(train_classifier)
    X2 = estimator.predict(test_images)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = scale_features(X, X_mean, X_std)
    X2 = scale_features(X2, X_mean, X_std)
    print "Transformed datasets"

    test_classifier(X, label_classifier, X2, T2)

    pylab.figure()
    pylab.subplots_adjust(wspace=0.0, hspace=0.0)
    n_cells = numpy.min((int(numpy.sqrt(n_filters)), 10))
    for i in range(n_cells**2):
        pylab.subplot(n_cells, n_cells, i + 1)
        pylab.imshow(estimator.C_[i].reshape(28, 28),
                     cmap=pylab.cm.gray, interpolation="nearest")
        pylab.xticks(())
        pylab.yticks(())
    pylab.show()
