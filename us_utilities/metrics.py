import numpy as np
import tensorflow as tf
import math
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans as KMeans
import jax

import tensorflow.keras.backend as K
EPS = 1.0e-7


def mutualInformation(bin_centers,
                      sigma_ratio=0.5,  # sigma for soft MI. If not provided, it will be half of a bin length
                      max_clip=1,
                      crop_background=False,  # crop_background should never be true if local_mi is True
                      local_mi=False,
                      patch_size=1):
    """
    mutual information for image-image pairs.
    Author: Courtney Guo. See thesis https://dspace.mit.edu/handle/1721.1/123142
    """
    print("vxm:mutual information loss is experimental.", file=sts.stderr)

    if local_mi:
        return localMutualInformation(bin_centers, sigma_ratio, max_clip, patch_size)

    else:
        return globalMutualInformation(bin_centers, sigma_ratio, max_clip, crop_background)


def globalMutualInformation(bin_centers,
                            sigma_ratio=0.5,
                            max_clip=1,
                            crop_background=False):
    """
    Mutual Information for image-image pairs
    Building from neuron.losses.MutualInformationSegmentation()
    This function assumes that y_true and y_pred are both (batch_size x height x width x depth x nb_chanels)
    Author: Courtney Guo. See thesis at https://dspace.mit.edu/handle/1721.1/123142
    """
    print("vxm:mutual information loss is experimental.", file=sts.stderr)

    """ prepare MI. """
    vol_bin_centers = K.variable(bin_centers)
    num_bins = len(bin_centers)
    sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

    preterm = K.variable(1 / (2 * np.square(sigma)))

    def mi(y_true, y_pred):
        """ soft mutual info """
        y_pred = K.clip(y_pred, 0, max_clip)
        y_true = K.clip(y_true, 0, max_clip)

        if crop_background:
            # does not support variable batch size
            thresh = 0.0001
            padding_size = 20
            filt = tf.ones([padding_size, padding_size, padding_size, 1, 1])

            smooth = tf.nn.conv3d(y_true, filt, [1, 1, 1, 1, 1], "SAME")
            mask = smooth > thresh
            # mask = K.any(K.stack([y_true > thresh, y_pred > thresh], axis=0), axis=0)
            y_pred = tf.boolean_mask(y_pred, mask)
            y_true = tf.boolean_mask(y_true, mask)
            y_pred = K.expand_dims(K.expand_dims(y_pred, 0), 2)
            y_true = K.expand_dims(K.expand_dims(y_true, 0), 2)

        else:
            # reshape: flatten images into shape (batch_size, heightxwidthxdepthxchan, 1)
            y_true = K.reshape(y_true, (-1, K.prod(K.shape(y_true)[1:])))
            y_true = K.expand_dims(y_true, 2)
            y_pred = K.reshape(y_pred, (-1, K.prod(K.shape(y_pred)[1:])))
            y_pred = K.expand_dims(y_pred, 2)

        nb_voxels = tf.cast(K.shape(y_pred)[1], tf.float32)

        # reshape bin centers to be (1, 1, B)
        o = [1, 1, np.prod(vol_bin_centers.get_shape().as_list())]
        vbc = K.reshape(vol_bin_centers, o)

        # compute image terms
        I_a = K.exp(- preterm * K.square(y_true - vbc))
        I_a /= K.sum(I_a, -1, keepdims=True)

        I_b = K.exp(- preterm * K.square(y_pred - vbc))
        I_b /= K.sum(I_b, -1, keepdims=True)

        # compute probabilities
        I_a_permute = K.permute_dimensions(I_a, (0, 2, 1))
        pab = K.batch_dot(I_a_permute, I_b)  # should be the right size now, nb_labels x nb_bins
        pab /= nb_voxels
        pa = tf.reduce_mean(I_a, 1, keep_dims=True)
        pb = tf.reduce_mean(I_b, 1, keep_dims=True)

        papb = K.batch_dot(K.permute_dimensions(pa, (0, 2, 1)), pb) + K.epsilon()
        mi = K.sum(K.sum(pab * K.log(pab / papb + K.epsilon()), 1), 1)

        return mi

    def loss(y_true, y_pred):
        return -mi(y_true, y_pred)

    return loss


def localMutualInformation(bin_centers,
                           vol_size,
                           sigma_ratio=0.5,
                           max_clip=1,
                           patch_size=1):
    """
    Local Mutual Information for image-image pairs
    # vol_size is something like (160, 192, 224)
    This function assumes that y_true and y_pred are both (batch_sizexheightxwidthxdepthxchan)
    Author: Courtney Guo. See thesis at https://dspace.mit.edu/handle/1721.1/123142
    """
    print("vxm:mutual information loss is experimental.", file=sts.stderr)

    """ prepare MI. """
    vol_bin_centers = K.variable(bin_centers)
    num_bins = len(bin_centers)
    sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

    preterm = K.variable(1 / (2 * np.square(sigma)))

    def local_mi(y_true, y_pred):
        y_pred = K.clip(y_pred, 0, max_clip)
        y_true = K.clip(y_true, 0, max_clip)

        # reshape bin centers to be (1, 1, B)
        o = [1, 1, 1, 1, num_bins]
        vbc = K.reshape(vol_bin_centers, o)

        # compute padding sizes
        x, y, z = vol_size
        x_r = -x % patch_size
        y_r = -y % patch_size
        z_r = -z % patch_size
        pad_dims = [[0, 0]]
        pad_dims.append([x_r // 2, x_r - x_r // 2])
        pad_dims.append([y_r // 2, y_r - y_r // 2])
        pad_dims.append([z_r // 2, z_r - z_r // 2])
        pad_dims.append([0, 0])
        padding = tf.constant(pad_dims)

        # compute image terms
        # num channels of y_true and y_pred must be 1
        I_a = K.exp(- preterm * K.square(tf.pad(y_true, padding, 'CONSTANT') - vbc))
        I_a /= K.sum(I_a, -1, keepdims=True)

        I_b = K.exp(- preterm * K.square(tf.pad(y_pred, padding, 'CONSTANT') - vbc))
        I_b /= K.sum(I_b, -1, keepdims=True)

        I_a_patch = tf.reshape(I_a, [(x + x_r) // patch_size, patch_size, (y + y_r) // patch_size, patch_size,
                                     (z + z_r) // patch_size, patch_size, num_bins])
        I_a_patch = tf.transpose(I_a_patch, [0, 2, 4, 1, 3, 5, 6])
        I_a_patch = tf.reshape(I_a_patch, [-1, patch_size ** 3, num_bins])

        I_b_patch = tf.reshape(I_b, [(x + x_r) // patch_size, patch_size, (y + y_r) // patch_size, patch_size,
                                     (z + z_r) // patch_size, patch_size, num_bins])
        I_b_patch = tf.transpose(I_b_patch, [0, 2, 4, 1, 3, 5, 6])
        I_b_patch = tf.reshape(I_b_patch, [-1, patch_size ** 3, num_bins])

        # compute probabilities
        I_a_permute = K.permute_dimensions(I_a_patch, (0, 2, 1))
        pab = K.batch_dot(I_a_permute, I_b_patch)  # should be the right size now, nb_labels x nb_bins
        pab /= patch_size ** 3
        pa = tf.reduce_mean(I_a_patch, 1, keep_dims=True)
        pb = tf.reduce_mean(I_b_patch, 1, keep_dims=True)

        papb = K.batch_dot(K.permute_dimensions(pa, (0, 2, 1)), pb) + K.epsilon()
        mi = K.mean(K.sum(K.sum(pab * K.log(pab / papb + K.epsilon()), 1), 1))

        return mi

    def loss(y_true, y_pred):
        return -local_mi(y_true, y_pred)

    return loss


def fit_kmeans_sklearn(image_data, n_clusters=5):
    org_shape = tf.shape(image_data)
    image_data = jax.numpy.array(image_data).reshape((-1, 1))
    print(f"DEVICE: {image_data.device_buffer.device()}")
    k_means = KMeans(init="k-means++", n_clusters=n_clusters, n_init=1, max_iter=5)
    k_means.fit(image_data)

    return
def fit_kmeans_tensor(image_data, k_means, iter=100):
    org_shape = tf.shape(image_data)
    att_to_km = np.array(image_data).reshape((-1, 1))
    def input_fn():
        data_tensor = tf.convert_to_tensor(att_to_km, dtype=tf.float32)

        return tf.compat.v1.train.limit_epochs(data_tensor,
                                               num_epochs=1)
    num_iterations = iter
    # previous_centers = None
    for _ in range(num_iterations):
        k_means.train(input_fn)
        cluster_centers = k_means.cluster_centers()
        # if previous_centers is not None:
        #     print('delta:', cluster_centers - previous_centers)
        previous_centers = cluster_centers
    #     print('score:', k_means.score(input_fn))
    # print('cluster centers:', cluster_centers)

    labels = k_means.predict_cluster_index(input_fn)
    cluster_centers = k_means.cluster_centers()

    return tf.reshape(labels, org_shape), cluster_centers

def fit_gmm(image_data, n_components=6):
    org_shape = image_data.shape
    image_data = image_data.reshape((-1, 1))
    gmm_model = GMM(n_components=n_components, covariance_type='full').fit(image_data)  # tied works better than full
    gmm_labels = gmm_model.predict(image_data)
    gmm_labels = gmm_labels.reshape(org_shape)

    return gmm_labels, gmm_model.means_


def separable_filter(tensor: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
    """
    Create a 3d separable filter.
    Here `tf.nn.conv3d` accepts the `filters` argument of shape
    (filter_depth, filter_height, filter_width, in_channels, out_channels),
    where the first axis of `filters` is the depth not batch,
    and the input to `tf.nn.conv3d` is of shape
    (batch, in_depth, in_height, in_width, in_channels).
    :param tensor: shape = (batch, dim1, dim2, dim3, 1)
    :param kernel: shape = (dim4,)
    :return: shape = (batch, dim1, dim2, dim3, 1)
    """
    strides = [1, 1, 1, 1, 1]
    kernel = tf.cast(kernel, dtype=tensor.dtype)

    tensor = tf.nn.conv3d(
        tf.nn.conv3d(
            tf.nn.conv3d(
                tensor,
                filters=tf.reshape(kernel, [-1, 1, 1, 1, 1]),
                strides=strides,
                padding="SAME",
            ),
            filters=tf.reshape(kernel, [1, -1, 1, 1, 1]),
            strides=strides,
            padding="SAME",
        ),
        filters=tf.reshape(kernel, [1, 1, -1, 1, 1]),
        strides=strides,
        padding="SAME",
    )
    return tensor


def rectangular_kernel1d(kernel_size: int) -> tf.Tensor:
    """
    Return a the 1D rectangular kernel for LocalNormalizedCrossCorrelation.
    :param kernel_size: scalar, size of the 1-D kernel
    :return: kernel_weights, of shape (kernel_size, )
    """

    kernel = tf.ones(shape=(kernel_size,), dtype=tf.float32)
    return kernel


def triangular_kernel1d(kernel_size: int) -> tf.Tensor:
    """
    Return a the 1D triangular kernel for LocalNormalizedCrossCorrelation.
    Assume kernel_size is odd, it will be a smoothed from
    a kernel which center part is zero.
    Then length of the ones will be around half kernel_size.
    The weight scale of the kernel does not matter as LNCC will normalize it.
    :param kernel_size: scalar, size of the 1-D kernel
    :return: kernel_weights, of shape (kernel_size, )
    """
    assert kernel_size >= 3
    assert kernel_size % 2 != 0

    padding = kernel_size // 2
    kernel = tf.constant(
        [0] * math.ceil(padding / 2)
        + [1] * (kernel_size - padding)
        + [0] * math.floor(padding / 2),
        dtype=tf.float32,
    )

    # (padding*2, )
    filters = tf.ones(shape=(kernel_size - padding, 1, 1), dtype=tf.float32)

    # (kernel_size, 1, 1)
    kernel = tf.nn.conv1d(
        kernel[None, :, None], filters=filters, stride=[1, 1, 1], padding="SAME"
    )

    return kernel[0, :, 0]


def gaussian_kernel1d(kernel_size: int) -> tf.Tensor:
    """
    Return a the 1D Gaussian kernel for LocalNormalizedCrossCorrelation.
    :param kernel_size: scalar, size of the 1-D kernel
    :return: filters, of shape (kernel_size, )
    """
    mean = (kernel_size - 1) / 2.0
    sigma = kernel_size / 3

    grid = tf.range(0, kernel_size, dtype=tf.float32)
    filters = tf.exp(-tf.square(grid - mean) / (2 * sigma ** 2))

    return filters


def gaussian_kernel1d_sigma(sigma: int) -> tf.Tensor:
    """
    Calculate a gaussian kernel.
    :param sigma: number defining standard deviation for
                  gaussian kernel.
    :return: shape = (dim, )
    """
    assert sigma > 0
    tail = int(sigma * 3)
    kernel = tf.exp([-0.5 * x ** 2 / sigma ** 2 for x in range(-tail, tail + 1)])
    kernel = kernel / tf.reduce_sum(kernel)
    return kernel


def cauchy_kernel1d(sigma: int) -> tf.Tensor:
    """
    Approximating cauchy kernel in 1d.
    :param sigma: int, defining standard deviation of kernel.
    :return: shape = (dim, )
    """
    assert sigma > 0
    tail = int(sigma * 5)
    k = tf.math.reciprocal([((x / sigma) ** 2 + 1) for x in range(-tail, tail + 1)])
    k = k / tf.reduce_sum(k)
    return k



def _hgram(img_l, img_r):
    hgram, _, _ = np.histogram2d(img_l.ravel(), img_r.ravel())
    return hgram


def _mutual_information(hgram):
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0

    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def mi(img_l, img_r):
    return _mutual_information(_hgram(img_l, img_r))

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, eps=1e-5, signed=False):
        self.win = win
        self.eps = eps
        self.signed = signed

    def ncc(self, Ii, Ji):
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(Ii.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims
        elif not isinstance(self.win, list):  # user specified a single number not a list
            self.win = [self.win] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        # compute filters
        in_ch = Ji.get_shape().as_list()[-1]
        sum_filt = tf.ones([*self.win, in_ch, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)

        # compute local sums via convolution
        padding = 'SAME'
        I_sum = conv_fn(Ii, sum_filt, strides, padding)
        J_sum = conv_fn(Ji, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win) * in_ch
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        # TODO: simplify this
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        cross = tf.maximum(cross, self.eps)
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var = tf.maximum(I_var, self.eps)
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var = tf.maximum(J_var, self.eps)

        if self.signed:
            cc = cross / tf.sqrt(I_var * J_var + self.eps)
        else:
            # cc = (cross * cross) / (I_var * J_var)
            cc = (cross / I_var) * (cross / J_var)

        return cc

    def loss(self, y_true, y_pred, reduce='mean'):
        # compute cc
        cc = self.ncc(y_true, y_pred)
        # reduce
        if reduce == 'mean':
            cc = tf.reduce_mean(K.batch_flatten(cc), axis=-1)
        elif reduce == 'max':
            cc = tf.reduce_max(K.batch_flatten(cc), axis=-1)
        elif reduce is not None:
            raise ValueError(f'Unknown NCC reduction type: {reduce}')
        # loss
        return -cc

class LocalNormalizedCrossCorrelation(tf.keras.losses.Loss):
    """
    Local squared zero-normalized cross-correlation.
    Denote y_true as t and y_pred as p. Consider a window having n elements.
    Each position in the window corresponds a weight w_i for i=1:n.
    Define the discrete expectation in the window E[t] as
        E[t] = sum_i(w_i * t_i) / sum_i(w_i)
    Similarly, the discrete variance in the window V[t] is
        V[t] = E[t**2] - E[t] ** 2
    The local squared zero-normalized cross-correlation is therefore
        E[ (t-E[t]) * (p-E[p]) ] ** 2 / V[t] / V[p]
    where the expectation in numerator is
        E[ (t-E[t]) * (p-E[p]) ] = E[t * p] - E[t] * E[p]
    Different kernel corresponds to different weights.
    For now, y_true and y_pred have to be at least 4d tensor, including batch axis.
    Reference:
        - Zero-normalized cross-correlation (ZNCC):
            https://en.wikipedia.org/wiki/Cross-correlation
        - Code: https://github.com/voxelmorph/voxelmorph/blob/legacy/src/losses.py
    """

    kernel_fn_dict = dict(
        gaussian=gaussian_kernel1d,
        rectangular=rectangular_kernel1d,
        triangular=triangular_kernel1d,
    )

    def __init__(
        self,
        kernel_size: int = 9,
        kernel_type: str = "rectangular",
        smooth_nr: float = EPS,
        smooth_dr: float = EPS,
        name: str = "LocalNormalizedCrossCorrelation",
        **kwargs,
    ):
        """
        Init.
        :param kernel_size: int. Kernel size or kernel sigma for kernel_type='gauss'.
        :param kernel_type: str, rectangular, triangular or gaussian
        :param smooth_nr: small constant added to numerator in case of zero covariance.
        :param smooth_dr: small constant added to denominator in case of zero variance.
        :param name: name of the loss.
        :param kwargs: additional arguments.
        """
        super().__init__(name=name, **kwargs)
        if kernel_type not in self.kernel_fn_dict.keys():
            raise ValueError(
                f"Wrong kernel_type {kernel_type} for LNCC loss type. "
                f"Feasible values are {self.kernel_fn_dict.keys()}"
            )
        self.kernel_fn = self.kernel_fn_dict[kernel_type]
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr

        # (kernel_size, )
        self.kernel = self.kernel_fn(kernel_size=self.kernel_size)
        # E[1] = sum_i(w_i), ()
        self.kernel_vol = tf.reduce_sum(
            self.kernel[:, None, None]
            * self.kernel[None, :, None]
            * self.kernel[None, None, :]
        )

    def calc_ncc(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return NCC for a batch.
        The kernel should not be normalized, as normalizing them leads to computation
        with small values and the precision will be reduced.
        Here both numerator and denominator are actually multiplied by kernel volume,
        which helps the precision as well.
        However, when the variance is zero, the obtained value might be negative due to
        machine error. Therefore a hard-coded clipping is added to
        prevent division by zero.
        :param y_true: shape = (batch, dim1, dim2, dim3, 1)
        :param y_pred: shape = (batch, dim1, dim2, dim3, 1)
        :return: shape = (batch, dim1, dim2, dim3. 1)
        """

        # t = y_true, p = y_pred
        # (batch, dim1, dim2, dim3, 1)
        t2 = y_true * y_true
        p2 = y_pred * y_pred
        tp = y_true * y_pred

        # sum over kernel
        # (batch, dim1, dim2, dim3, 1)
        t_sum = separable_filter(y_true, kernel=self.kernel)  # E[t] * E[1]
        p_sum = separable_filter(y_pred, kernel=self.kernel)  # E[p] * E[1]
        t2_sum = separable_filter(t2, kernel=self.kernel)  # E[tt] * E[1]
        p2_sum = separable_filter(p2, kernel=self.kernel)  # E[pp] * E[1]
        tp_sum = separable_filter(tp, kernel=self.kernel)  # E[tp] * E[1]

        # average over kernel
        # (batch, dim1, dim2, dim3, 1)
        t_avg = t_sum / self.kernel_vol  # E[t]
        p_avg = p_sum / self.kernel_vol  # E[p]

        # shape = (batch, dim1, dim2, dim3, 1)
        cross = tp_sum - p_avg * t_sum  # E[tp] * E[1] - E[p] * E[t] * E[1]
        t_var = t2_sum - t_avg * t_sum  # V[t] * E[1]
        p_var = p2_sum - p_avg * p_sum  # V[p] * E[1]

        # ensure variance >= 0
        t_var = tf.maximum(t_var, 0)
        p_var = tf.maximum(p_var, 0)

        # (E[tp] - E[p] * E[t]) ** 2 / V[t] / V[p]
        ncc = (cross * cross + self.smooth_nr) / (t_var * p_var + self.smooth_dr)

        return ncc

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.
        TODO: support channel axis dimension > 1.
        :param y_true: shape = (batch, dim1, dim2, dim3)
            or (batch, dim1, dim2, dim3, 1)
        :param y_pred: shape = (batch, dim1, dim2, dim3)
            or (batch, dim1, dim2, dim3, 1)
        :return: shape = (batch,)
        """
        # sanity checks
        if len(y_true.shape) == 4:
            y_true = tf.expand_dims(y_true, axis=4)
        if y_true.shape[4] != 1:
            raise ValueError(
                "Last dimension of y_true is not one. " f"y_true.shape = {y_true.shape}"
            )
        if len(y_pred.shape) == 4:
            y_pred = tf.expand_dims(y_pred, axis=4)
        if y_pred.shape[4] != 1:
            raise ValueError(
                "Last dimension of y_pred is not one. " f"y_pred.shape = {y_pred.shape}"
            )

        ncc = self.calc_ncc(y_true=y_true, y_pred=y_pred)
        return tf.reduce_mean(ncc, axis=[1, 2, 3, 4])

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config.update(
            kernel_size=self.kernel_size,
            kernel_type=self.kernel_type,
            smooth_nr=self.smooth_nr,
            smooth_dr=self.smooth_dr,
        )
        return config