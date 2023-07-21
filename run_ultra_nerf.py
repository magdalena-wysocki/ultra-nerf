import os
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import imageio
import time
import tensorflow_probability as tfp
from run_nerf_helpers import *
from load_us import load_us_data
from us_utilities import metrics
from tensorflow.keras import backend as K

tf.compat.v1.enable_eager_execution()


# TODO: Improve psf shape. Now, it is a 2D Gaussian kernel.
def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                    ):
    delta_t = 1  # 9.197324e-01
    x_cos = np.array(list(range(-size, size + 1)), dtype=np.float32)
    x_cos *= delta_t
    d1 = tf.distributions.Normal(mean, std * 2.)
    d2 = tf.distributions.Normal(mean, std)
    vals_x = d1.prob(tf.range(start=-size, limit=(size + 1), dtype=tf.float32) * delta_t)
    vals_y = d2.prob(tf.range(start=-size, limit=(size + 1), dtype=tf.float32) * delta_t)

    gauss_kernel = tf.einsum('i,j->ij',
                             vals_x,
                             vals_y)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)


# TODO: Change to args
g_size = 3
g_mean = 0.
g_variance = 1.
g_kernel = gaussian_kernel(g_size, g_mean, g_variance)
g_kernel = tf.constant(g_kernel[:, :, tf.newaxis, tf.newaxis], dtype=tf.float32)


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return tf.concat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, fn, embed_fn, netchunk=512 * 32):
    """Prepares inputs and applies network 'fn'."""
    fn.run_eagerly = True
    inputs_flat = tf.reshape(inputs, [-1, inputs.shape[-1]])

    embedded = embed_fn(inputs_flat)
    outputs_flat = batchify(fn, netchunk)(embedded)

    outputs = tf.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])

    return outputs


def render_method_convolutional_ultrasound(raw, z_vals, args):
    def raw2attenualtion(raw, dists):
        return tf.exp(-raw * dists)

    # Compute distance between points
    # In paper the points are sampled equidistantly
    dists = tf.math.abs(z_vals[..., :-1, None] - z_vals[..., 1:, None])
    dists = tf.squeeze(dists)
    dists = tf.concat([dists, dists[:, -1, None]], axis=-1)
    # ATTENUATION
    # Predict attenuation coefficient for each sampled point. This value is positive.
    attenuation_coeff = tf.math.abs(raw[..., 0])
    attenuation = raw2attenualtion(attenuation_coeff, dists)
    # Compute total attenuation at each pixel location as exp{-sum[a_n*d_n]}
    attenuation_transmission = tf.math.cumprod(attenuation, axis=1, exclusive=True)
    # REFLECTION
    prob_border = tf.math.sigmoid(raw[..., 2])

    # Bernoulli distribution can be approximated by RelaxedBernoulli
    # temperature = 0.01
    # border_distribution = tf.contrib.distributions.RelaxedBernoulli(temperature, probs=prob_border)
    # Note: Estimating a border explicitly is not necessary. I recommend experimenting with solely relying on
    # reflection coefficient for the geometry estimation
    border_distribution = tf.contrib.distributions.Bernoulli(probs=prob_border, dtype=tf.float32)
    border_indicator = tf.stop_gradient(border_distribution.sample(seed=0))
    # Predict reflection coefficient. This value is between (0, 1).
    reflection_coeff = tf.math.sigmoid(raw[..., 1])
    reflection_transmission = 1. - reflection_coeff * border_indicator
    reflection_transmission = tf.math.cumprod(reflection_transmission, axis=1, exclusive=True)
    reflection_transmission = tf.squeeze(reflection_transmission)
    border_convolution = tf.nn.conv2d(input=border_indicator[tf.newaxis, :, :, tf.newaxis], filter=g_kernel,
                                      strides=1, padding="SAME")
    border_convolution = tf.squeeze(border_convolution)

    # BACKSCATTERING
    # Scattering density coefficient can be either learned or constant for fully developed speckle
    density_coeff_value = tf.math.sigmoid(raw[..., 3])
    density_coeff = tf.ones_like(reflection_coeff) * density_coeff_value
    scatter_density_distibution = tfp.distributions.Bernoulli(probs=density_coeff, dtype=tf.float32)
    scatterers_density = scatter_density_distibution.sample()
    # Predict scattering amplitude
    amplitude = tf.math.sigmoid(raw[..., 4])
    # Compute scattering template
    scatterers_map = tf.math.multiply(scatterers_density, amplitude)
    psf_scatter = tf.nn.conv2d(input=scatterers_map[tf.newaxis, :, :, tf.newaxis], filter=g_kernel, strides=1,
                               padding="SAME")
    psf_scatter = tf.squeeze(psf_scatter)
    # Compute remaining intensity at a point n
    transmission = tf.math.multiply(attenuation_transmission, reflection_transmission)
    # Compute backscattering part of the final echo
    b = tf.math.multiply(transmission, psf_scatter)
    # Compute reflection part of the final echo
    r = tf.math.multiply(tf.math.multiply(transmission, reflection_coeff), border_convolution)
    # Compute the final echo
    # Note: log compression has not been used for the submission
    # if args.log_compression:
    #     compression_constant = 3.14  # TODO: should be calculated based on r_reflection_maximum
    #     log_compression = lambda x: tf.math.log(1. + compression_constant * x) * tf.math.log(
    #         1. + compression_constant)
    #     r = log_compression(r)
    intensity_map = b + r
    ret = {'intensity_map': intensity_map,
           'attenuation_coeff': attenuation_coeff,
           'reflection_coeff': reflection_coeff,
           'attenuation_transmission': attenuation_transmission,
           'reflection_transmission': reflection_transmission,
           'scatterers_density': scatterers_density,
           'scatterers_density_coeff': density_coeff,
           'scatter_amplitude': amplitude,
           'b': b,
           'r': r,
           "transmission": transmission}
    return ret


def render_rays_us(ray_batch,
                   network_fn,
                   network_query_fn,
                   N_samples,
                   retraw=False,
                   lindisp=False,
                   args=None):
    """Volumetric rendering.

        Args:
          ray_batch: array of shape [batch_size, ...]. We define rays and do not sample.

        Returns:

        """

    def raw2outputs(raw, z_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values.
        """
        ret = render_method_convolutional_ultrasound(raw, z_vals, args)
        return ret

    ###############################
    # batch size
    N_rays = ray_batch.shape[0]

    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction.
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance.
    bounds = tf.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # Decide where to sample along each ray. Under the logic, all rays will be sampled at
    # the same times.
    t_vals = tf.linspace(0., 1., N_samples)
    if not lindisp:
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
    z_vals = tf.broadcast_to(z_vals, [N_rays, N_samples])

    # Points in space to evaluate model at.
    origin = rays_o[..., None, :]
    step = rays_d[..., None, :] * \
           z_vals[..., :, None]

    pts = step + origin

    # Evaluate model at each point.
    raw = network_query_fn(pts, network_fn)  # [N_rays, N_samples, 5]
    ret = raw2outputs(
        raw, z_vals, rays_d)

    if retraw:
        ret['raw'] = raw

    for k in ret:
        tf.debugging.check_numerics(ret[k], 'output {}'.format(k))

    return ret


def batchify_rays(rays_flat, c2w=None, chunk=32 * 256, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays_us(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: tf.concat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render_us(H, W, sw, sh,
              chunk=1024 * 32, rays=None, c2w=None,
              near=0., far=55. * 0.001,
              **kwargs):
    """Render rays
    """

    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays_us_linear(H, W, sw, sh, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32)
    rays_d = tf.cast(tf.reshape(rays_d, [-1, 3]), dtype=tf.float32)
    near, far = near * \
                tf.ones_like(rays_d[..., :1]), far * tf.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = tf.concat([rays_o, rays_d, near, far], axis=-1)

    # Render and reshape
    all_ret = batchify_rays(rays, c2w=c2w, chunk=chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = tf.reshape(all_ret[k], k_sh)
    return all_ret


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(render_poses):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(
            H, W, focal, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.numpy())
        disps.append(disp.numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            print(p)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model."""
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed, args.i_embed_gauss)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    output_ch = args.output_ch
    skips = [4]

    model = init_nerf_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)

    grad_vars = model.trainable_variables
    models = {'model': model}

    # We sample points equidistantly at the pixel location.
    # TODO: After sampling along a ray at any point use fine model
    # model_fine = None
    # if args.N_importance > 0:
    #     model_fine = init_nerf_model(
    #         D=args.netdepth_fine, W=args.netwidth_fine,
    #         input_ch=input_ch, output_ch=output_ch, skips=skips,
    #         input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    #     grad_vars += model_fine.trainable_variables
    #     models['model_fine'] = model_fine

    def network_query_fn(inputs, network_fn):
        return run_network(
            inputs, network_fn,
            embed_fn=embed_fn,
            netchunk=args.netchunk)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'N_samples': args.N_samples,
        'network_fn': model
    }

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}

    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f)]
    print('Found ckpts', ckpts)

    if len(ckpts) > 0 and not args.no_reload:
        ft_weights = ckpts[-1]
        print('Reloading from', ft_weights)
        model.set_weights(np.load(ft_weights, allow_pickle=True))
        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to', start)
        #
        # if model_fine is not None:
        #     ft_weights_fine = '{}_fine_{}'.format(
        #         ft_weights[:-11], ft_weights[-10:])
        #     print('Reloading fine from', ft_weights_fine)
        #     model_fine.set_weights(np.load(ft_weights_fine, allow_pickle=True))

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path', default='config_fern.txt')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str,
                        default='./data/llff/fern', help='input data directory')

    # training options
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument("--ssim_filter_size", type=int, default=7)
    parser.add_argument("--ssim_lambda", type=float, default=0.75)
    parser.add_argument("--loss", type=str, default='l2')
    parser.add_argument('--probe_depth', type=int, default=140)
    parser.add_argument('--probe_width', type=int, default=80)
    parser.add_argument("--output_ch", type=int, default=5),
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=128,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int,
                        default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=128,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--chunk", type=int, default=4096 * 16,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=4096 * 16,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--random_seed", type=int, default=None,
                        help='fix random seed for repeatability')

    # pre-crop options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--i_embed_gauss", type=int, default=0,
                        help='mapping size for Gaussian positional encoding, 0 for none')

    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='us',
                        help='options: us')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=50,
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_img", type=int, default=100,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=100,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=5000000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=5000000,
                        help='frequency of render_poses video saving')

    parser.add_argument("--log_compression", action='store_true')
    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()

    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    # Load data
    if args.dataset_type == 'us':
        images, poses, i_test = load_us_data(args.datadir)

        if not isinstance(i_test, list):
            i_test = [i_test]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print("Test {}, train {}".format(len(i_test), len(i_train)))

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # The poses are not normalized. We scale down the space.
    # It is possible to normalize poses and remove scaling.
    scaling = 0.001
    near = 0
    probe_depth = args.probe_depth * scaling
    probe_width = args.probe_width * scaling
    far = probe_depth
    H, W = images.shape[1], images.shape[2]
    sy = probe_depth / float(H)
    sx = probe_width / float(W)
    sh = sy
    sw = sx
    # H, W = int(H), int(W)

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(
        args)

    bds_dict = {
        'near': tf.cast(near, tf.float32),
        'far': tf.cast(far, tf.float32),
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    render_kwargs_train["args"] = args

    # Create optimizer
    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
                                                               decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)
    models['optimizer'] = optimizer
    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(start)

    N_iters = args.n_iters
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    # Summary writers
    writer = tf.contrib.summary.create_file_writer(
        os.path.join(basedir, 'summaries', expname))
    writer.set_as_default()

    for i in range(start, N_iters):
        time0 = time.time()
        # Sample random ray batch
        # Random from one image
        img_i = np.random.choice(i_train)
        try:
            target = tf.transpose(images[img_i])
        except:
            print(img_i)

        pose = poses[img_i, :3, :4]
        ssim_weight = args.ssim_lambda
        l2_weight = 1. - ssim_weight

        rays_o, rays_d = get_rays_us_linear(H, W, sw, sh, pose)
        batch_rays = tf.stack([rays_o, rays_d], 0)
        loss = dict()
        loss_holdout = dict()
        #####  Core optimization loop  #####
        with tf.GradientTape() as tape:
            # Make predictions
            rendering_output = render_us(
                H, W, sw, sh, c2w=pose, chunk=args.chunk, rays=batch_rays,
                retraw=True, **render_kwargs_train)

            output_image = rendering_output['intensity_map']
            if args.loss == 'l2':
                l2_intensity_loss = img2mse(output_image, target)
                loss["l2"] = (1., l2_intensity_loss)
            elif args.loss == 'ssim':
                ssim_intensity_loss = 1. - tf.image.ssim_multiscale(tf.expand_dims(tf.expand_dims(output_image, 0), -1),
                                                                    tf.expand_dims(tf.expand_dims(target, 0), -1),
                                                                    max_val=1.0, filter_size=args.ssim_filter_size,
                                                                    filter_sigma=1.5, k1=0.01, k2=0.1
                                                                    )
                loss["ssim"] = (ssim_weight, ssim_intensity_loss)
                l2_intensity_loss = img2mse(output_image, target)
                loss["l2"] = (l2_weight, l2_intensity_loss)

            total_loss = 0.
            for loss_value in loss.values():
                total_loss += loss_value[0] * loss_value[1]

        gradients = tape.gradient(total_loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))
        dt = time.time() - time0

        #####           end            #####

        # Rest is logging
        def save_weights(net, prefix, i):
            path = os.path.join(
                basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
            np.save(path, net.get_weights())
            print('saved weights at', path)

        if i % args.i_weights == 0:
            for k in models:
                save_weights(models[k], k, i)

        if i % args.i_print == 0 or i < 10:
            print(expname, i, total_loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))
            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                g_i = 0
                for t in gradients:
                    g_i += 1
                    tf.contrib.summary.histogram(str(g_i), t)
                tf.contrib.summary.scalar('misc/learning_rate', K.eval(optimizer.learning_rate(optimizer.iterations)))
                loss_string = "Total loss = "
                for l_key, l_value in loss.items():
                    loss_string += f' + {l_value[0]} * {l_key}'
                    tf.contrib.summary.scalar(f'train/loss_{l_key}/', l_value[1])
                    tf.contrib.summary.scalar(f'train/penalty_factor_{l_key}/', l_value[0])
                    tf.contrib.summary.scalar(f'train/total_loss_{l_key}/', l_value[0] * l_value[1])
                tf.contrib.summary.scalar('train/total_loss/', total_loss)
                print(loss_string)
            if i % args.i_img == 0:
                # Log a rendered validation view to Tensorboard
                img_i = np.random.choice(i_val)
                target = tf.transpose(images[img_i])
                pose = poses[img_i, :3, :4]
                rendering_output_test = render_us(H, W, sw, sh, chunk=args.chunk, c2w=pose,
                                                  **render_kwargs_test)

                # TODO: Duplicaetes the loss calculation. Should be a function.
                output_image_test = rendering_output_test['intensity_map']
                if args.loss == 'l2':
                    l2_intensity_loss = img2mse(output_image_test, target)
                    loss_holdout["l2"] = (1., l2_intensity_loss)
                elif args.loss == 'ssim':
                    ssim_intensity_loss = 1. - tf.image.ssim_multiscale(
                        tf.expand_dims(tf.expand_dims(output_image_test, 0), -1),
                        tf.expand_dims(tf.expand_dims(target, 0), -1),
                        max_val=1.0, filter_size=args.ssim_filter_size,
                        filter_sigma=1.5, k1=0.01, k2=0.1
                    )
                    loss_holdout["ssim"] = (ssim_weight, ssim_intensity_loss)
                    l2_intensity_loss = img2mse(output_image_test, target)
                    loss_holdout["l2"] = (l2_weight, l2_intensity_loss)

                total_loss_holdout = 0.
                for loss_value in loss_holdout.values():
                    total_loss_holdout += loss_value[0] * loss_value[1]

                # Save out the validation image for Tensorboard-free monitoring
                testimgdir = os.path.join(basedir, expname, 'tboard_val_imgs')
                # if i==0:
                os.makedirs(testimgdir, exist_ok=True)
                imageio.imwrite(os.path.join(testimgdir,
                                             '{:06d}.png'.format(i)), to8b(tf.transpose(output_image_test)))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('b_mode/output/',
                                             tf.expand_dims(tf.expand_dims(to8b(tf.transpose(output_image_test)), 0),
                                                            -1))
                    for l_key, l_value in loss_holdout.items():
                        tf.contrib.summary.scalar(f'test/loss_{l_key}/', l_value[0])
                        tf.contrib.summary.scalar(f'test/penalty_factor_{l_key}/', l_value[1])
                        tf.contrib.summary.scalar(f'test/total_loss_{l_key}/', l_value[0] * l_value[1])
                    tf.contrib.summary.scalar('test/total_loss/', total_loss)
                    tf.contrib.summary.image('b_mode/target/',
                                             tf.expand_dims(tf.expand_dims(to8b(tf.transpose(target)), 0), -1))
                    for map_k, map_v in rendering_output_test.items():
                        tf.contrib.summary.image(f'maps/{map_k}/',
                                                 tf.expand_dims(tf.image.decode_png(
                                                     show_colorbar(tf.transpose(map_v)).getvalue(), channels=4),
                                                     0))

        global_step.assign_add(1)


if __name__ == '__main__':
    train()
