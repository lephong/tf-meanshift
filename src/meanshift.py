import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

n_samples = 300 # sample size
n_centroids = 10 # number of centroids
n_updates = 30 # number of updates

n_gaussians = 3 # 'true' number of clusters

window_radius = .1
dim = 2
input_X = tf.placeholder(tf.float32, [None, dim])
init_C = tf.placeholder(tf.float32, [None, dim])

colors = ['r', 'c', 'm', 'g', 'k'] * 100


def create_data(n_points, n_components):
    coefs = np.random.rand(n_components)
    coefs = coefs / np.sum(coefs)

    # create mean and cov
    mean = np.random.rand(n_components, dim)
    var = np.random.rand(n_components, dim, dim)

    for i in range(n_components):
        var[i, :, :] = np.diag(np.random.rand(dim)) / 30

    # sampling
    ids = np.random.choice(list(range(n_components)), size=n_points, replace=True, p=coefs)
    points = []
    for i in range(n_components):
        n = np.sum((ids == i).astype(np.int32))
        points.append(np.random.multivariate_normal(mean=mean[i, :], cov=var[i, :, :], size=n))

    return points


def mean_shift(n_updates):
    X1 = tf.expand_dims(tf.transpose(input_X), 0)
    X2 = tf.expand_dims(input_X, 0)
    C = init_C

    sbs_C = [init_C]

    def _mean_shift_step(C):
        C = tf.expand_dims(C, 2)
        Y = tf.reduce_sum(tf.pow((C - X1) / window_radius, 2), axis=1)
        gY = tf.exp(-Y)
        num = tf.reduce_sum(tf.expand_dims(gY, 2) * X2, axis=1)
        denom = tf.reduce_sum(gY, axis=1, keep_dims=True)
        C = num / denom
        return C

    for i in range(n_updates):
        C = _mean_shift_step(C)
        sbs_C.append(C)

    return C, tf.pack(sbs_C)


def plot(X, C, sbs_C):
    for i, ps in enumerate(X):
        plt.plot(ps[:, 0], ps[:, 1], colors[i] + '.')

    for i in range(sbs_C.shape[1]):
        plt.plot(sbs_C[:, i, 0], sbs_C[:, i, 1], 'y')

    plt.plot(C[:, 0], C[:, 1], 'bo')

    plt.show()


if __name__ == "__main__":
    X = create_data(n_samples, n_gaussians)
    stacked_X = np.vstack(X)
    C = stacked_X[np.random.randint(stacked_X.shape[0], size=n_centroids), :]
    ms_C, sbs_ms_C = mean_shift(n_updates)

    sess = tf.Session()
    t = time.process_time()
    C, sbs_C = sess.run([ms_C, sbs_ms_C], feed_dict={input_X: stacked_X, init_C: C})
    print(time.process_time() - t)

    sbs_C = np.reshape(sbs_C, [n_updates + 1, C.shape[0], C.shape[1]])

    plot(X, C, sbs_C)



