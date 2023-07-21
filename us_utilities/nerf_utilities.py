from run_nerf_helpers import *

tf.compat.v1.enable_eager_execution()
import jax.numpy as jnp


def patch_l2(y, y_prim):
    patches_y = tf.image.extract_patches(y[tf.newaxis, :, :, tf.newaxis], [1, 8, 8, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')
    patches_y_prim = tf.image.extract_patches(y_prim[tf.newaxis, :, :, tf.newaxis], [1, 8, 8, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')
    patch_sum_y = tf.reduce_sum(patches_y, axis=3, keepdims=True)
    patch_sum_y_prim = tf.reduce_sum(patches_y_prim, axis=3, keepdims=True)

    img2mse(tf.squeeze(patch_sum_y), tf.squeeze(patch_sum_y_prim))


def compute_tv_norm(values, losstype='l2', weighting=None):  # pylint: disable=g-doc-args
    """Returns TV norm for input values.
    Note: The weighting / masking term was necessary to avoid degenerate
    solutions on GPU; only observed on individual DTU scenes.
    """
    v00 = values[ :-1, :-1]
    v01 = values[ :-1, 1:]
    v10 = values[ 1:, :-1]

    if losstype == 'l2':
        loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
    elif losstype == 'l1':
        loss = jnp.abs(v00 - v01) + jnp.abs(v00 - v10)
    else:
        raise ValueError('Not supported losstype.')

    if weighting is not None:

        loss = loss * weighting
    return loss

def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                    ):
    delta_t = 1  # 9.197324e-01
    x_cos = np.array(list(range(-size, size +1)), dtype=np.float32)
    x_cos *= delta_t

    y_modulation = tf.cos(x_cos * 2 *np.pi *8e6)
    d1 = tf.distributions.Normal(mean, std *3.)

    d2 = tf.distributions.Normal(mean, std)
    vals_x = d1.prob(tf.range(start=-size, limit=(size + 1), dtype=tf.float32 ) *delta_t)
    vals_y = d2.prob(tf.range(start=-size, limit=(size + 1), dtype=tf.float32 ) *delta_t)

    gauss_kernel = tf.einsum('i,j->ij',
                             vals_x,
                             vals_y)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)