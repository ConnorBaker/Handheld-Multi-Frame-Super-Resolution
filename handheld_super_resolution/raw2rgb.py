import random
import math
import warnings

import exifread
import numpy as np
from skimage import img_as_float32, filters

import cv2

def get_xyz2cam_from_exif(impath):
    # Open image file for reading (must be in binary mode)
    f = open(impath, 'rb')

    # Return the exif tags
    tags = exifread.process_file(f)

    # Get the 9 values of the first CCM in the EXIF
    color_matrix1 = tags['Image Tag 0xC621']
    color_matrix1 = np.array([x.decimal() for x in color_matrix1.values])


    # Fill the matrix and return
    xyz2cam = np.reshape(color_matrix1, (3, 3))

    return xyz2cam.astype(np.float32)



def get_random_ccm():
    """Generates random RGB -> Camera color correction matrices."""
    # Takes a random convex combination of XYZ -> Camera CCMs.
    xyz2cams = [[[1.0234, -0.2969, -0.2266],
               [-0.5625, 1.6328, -0.0469],
               [-0.0703, 0.2188, 0.6406]],
              [[0.4913, -0.0541, -0.0202],
               [-0.613, 1.3513, 0.2906],
               [-0.1564, 0.2151, 0.7183]],
              [[0.838, -0.263, -0.0639],
               [-0.2887, 1.0725, 0.2496],
               [-0.0627, 0.1427, 0.5438]],
              [[0.6596, -0.2079, -0.0562],
               [-0.4782, 1.3016, 0.1933],
               [-0.097, 0.1581, 0.5181]]]

    num_ccms = len(xyz2cams)  # (4,3,3)

    weights = np.random.rand(num_ccms).reshape((num_ccms, 1, 1))
    weights_sum = weights.sum()
    xyz2cam = (xyz2cams * weights).sum(axis=0) / weights_sum

    # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    rgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]])
    rgb2cam = xyz2cam @ rgb2xyz

    # Normalizes each row.
    rgb2cam = rgb2cam / rgb2cam.sum(axis=-1, keepdim=True)
    return rgb2cam


def get_random_noise_parameters(log_min_shot=0.0001, log_max_shot=0.012, sigma_read_noise=0.26):
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = math.log(log_min_shot)
    log_max_shot_noise = math.log(log_max_shot)
    log_shot_noise = random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = math.exp(log_shot_noise)

    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + random.gauss(mu=0.0, sigma=sigma_read_noise)
    read_noise = math.exp(log_read_noise)
    return shot_noise, read_noise


def get_random_gains():
    """Generates random gains for brightening and white balance."""
    # RGB gain represents brightening.
    rgb_gain = 1.0 / random.gauss(mu=0.8, sigma=0.1)

    # Red and blue gains represent white balance.
    red_gain = random.uniform(1.9, 2.4)
    blue_gain = random.uniform(1.5, 1.9)
    return rgb_gain, red_gain, blue_gain


def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
    """Inverts gains while safely handling saturated pixels."""
    assert image.ndim == 3 and image.shape[2] == 3

    gains = np.array([1.0 / red_gain, 1.0, 1.0 / blue_gain]) / rgb_gain
    gains = gains.reshape((1, 1, 3))

    # Prevents dimming of saturated pixels by smoothly masking gains near white.
    gray = np.mean(image, axis=-1, keepdims=True)
    inflection = 0.9
    mask = ((gray - inflection).cllp(min=0.0) / (1.0 - inflection))
    mask = mask * mask

    safe_gains = np.max(mask + (1.0 - mask) * gains, gains)
    return image * safe_gains


def apply_gains(image, red_gain, blue_gain, rgb_gain):
    """Inverts gains while safely handling saturated pixels."""
    assert image.ndim == 3 and image.shape[-1] in [3, 4]

    if image.shape[-1] == 3:
        gains = np.tensor([red_gain, 1.0, blue_gain]) * rgb_gain
    else:
        gains = np.tensor([red_gain, 1.0, 1.0, blue_gain]) * rgb_gain
    return (image * gains).clip(a_min=0.0, a_max=1.0)


def get_color_matrix(raw, xyz2cam=None):
    rgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
    # If xyz2cam is not given, take it from rawpy.
    if xyz2cam is None:
        xyz2cam = raw.rgb_xyz_matrix[:3]
    if np.linalg.norm(xyz2cam) == 0:
        print('Warning -- CCM not found or given. Use eye matrix instead.')
        rgb2cam = rgb2xyz
    else:        
        rgb2cam = xyz2cam @ rgb2xyz

    # Normalizes each row.
    rgb2cam = rgb2cam / rgb2cam.sum(axis=-1, keepdims=True)
    return rgb2cam.astype(np.float32)


def apply_ccm(image, ccm):
    assert(image.ndim == 3 and image.shape[-1] == 3)
    image = np.transpose(image, (2, 0, 1))
    shape = image.shape
    image = image.reshape(3, -1)
    image = np.matmul(ccm, image)
    image = image.reshape(shape)
    return np.transpose(image, (1, 2, 0))


def gamma_compression(img, gamma=2.2):
    img = np.clip(img, a_min=0.0, a_max=1.0)
    return img**(1./gamma)


def gamma_expansion(img, gamma=2.2):
    img = np.clip(img, a_min=1e-8, a_max=1.0)
    return img ** gamma


def apply_smoothstep(image):
    """Apply global tone mapping curve."""
    # image_out = 3 * image**2 - 2 * image**3
    
    # tonemap = cv2.createTonemap(1.0)
    # image_out = tonemap.process(image)
    
    from skimage import img_as_ubyte, img_as_float32
    times = [1, 0.5, 2]
    images = [img_as_ubyte(np.clip(image*i, 0, 1)) for i in times] 
    
    
    merge_mertens = cv2.createMergeMertens()
    image_out = merge_mertens.process(images)
    image_out = img_as_float32(image_out)

    image_out = 3 * image_out**2 - 2 * image_out**3
    return image_out


def invert_smoothstep(image):
    """Approximately inverts a global tone mapping curve."""
    image = np.clip(image, a_min=0.0, a_max=1.0)
    return 0.5 - np.sin(np.arcsin(1.0 - 2.0 * image) / 3.0)


def unprocess_isp(jpg, log_max_shot=0.012):
    """
    Convert a jpg image to raw image.
    """
    rgb2cam = get_random_ccm()
    cam2rgb = np.linalg.inv(rgb2cam)
    rgb_gain, red_gain, blue_gain = get_random_gains()
    lambda_read, lambda_shot = get_random_noise_parameters(log_max_shot=log_max_shot)
    metadata = {'rgb2cam': rgb2cam, 'cam2rgb': cam2rgb, 'rgb_gain': rgb_gain, 'red_gain': red_gain,
                'blue_gain': blue_gain, 'lambda_shot': lambda_shot, 'lambda_read': lambda_read}

    ## Inverse tone mapping
    jpg = invert_smoothstep(jpg)

    ## Gamma expansion
    jpg = gamma_expansion(jpg)

    ## Inverse color matrix
    raw = apply_ccm(jpg, rgb2cam)

    ## Inverse gains
    raw = safe_invert_gains(raw, red_gain, blue_gain, rgb_gain)

    return raw, metadata

def devignette(image):
    h, w, _ = image.shape
    vignette_filter = np.abs(np.linspace(-h/w * np.pi/2, h/w * np.pi/2, h))
    vignette_filter = np.outer(vignette_filter, np.abs(np.linspace(-np.pi/2, np.pi/2, w)))
    
    image_out = (2 - np.cos(vignette_filter)**4)[:,:,None] * image 
    return image_out

def _deblur(img, deblur_config):
    """Polyblur blind deblurring of a linear-RGB image (numpy HWC in [0, 1]).

    Replaces the unsharp mask when enabled: Polyblur estimates a small
    anisotropic-Gaussian blur and inverts it with a polynomial of the blur
    operator (no iterative deconvolution), targeting exactly the ~1px optical /
    kernel-regression softness. Runs on the GPU via ``torch.fft`` and returns
    numpy HWC. See the vendored ``polyblur`` package (MIT, teboli/polyblur).
    """
    import torch
    from .polyblur import polyblur_deblurring

    t = torch.as_tensor(np.ascontiguousarray(img), dtype=torch.float32, device="cuda")
    t = t.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)  # (1, 3, H, W)
    # prefiltering is SNR-resolved (params.py) to a bool; if it arrives unresolved
    # (e.g. a raw config), default to the noise-safe True.
    pf = deblur_config.get("prefiltering", True)
    out = polyblur_deblurring(
        t,
        n_iter=int(deblur_config.get("n_iter", 1)),
        c=float(deblur_config.get("c", 0.352)),       # blur-model slope
        b=float(deblur_config.get("sigma_b", 0.768)),  # blur-model intercept (higher => less blur estimated)
        alpha=float(deblur_config.get("alpha", 2)),
        beta=float(deblur_config.get("beta", 3)),
        remove_halo=bool(deblur_config.get("remove_halo", True)),
        prefiltering=True if pf == "SNR_based" else bool(pf),
        edgetaping=bool(deblur_config.get("edgetaping", True)),
    )
    return out.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()


def _measured_rl_deblur(img, cfg, edge_strength=None):
    """Measured-PSF Richardson-Lucy deblur of a linear-RGB image (numpy HWC in
    [0, 1]), GATED by the merge's structure-tensor edge-strength field.

    Opt-in alternative to the aberration stage's BOUNDED, blind Polyblur deblur
    (selected by ``postprocessing.aberration.deblur_method == 'measured_rl'``).
    Polyblur caps high-frequency gain at ``beta`` and its blind sigma-estimate is
    fooled by high-contrast edges; RL is unbounded, so given a MEASURED Gaussian
    PSF (``sigma`` -- a per-camera/scale calibration value, like ``sigma_b``, NOT a
    blind per-tile estimate) it can target the merge's residual softness.

    ``edge_strength`` is the pipeline's OWN ``l1`` field (the merge's structure
    tensor, exposed via ``merging.expose_edge_strength`` ->
    ``debug_dict['edge_strength']``), at this image's resolution. It is turned into
    a [0,1] gate by ``deconv.edge_gate_from_l1`` and the sharpening is blended in
    only at edges -- ``out = img + gate*(deconv - img)`` -- so RL does not amplify
    flat-region noise. With no ``edge_strength`` the deconv is applied unblended
    (gate == 1) and a warning is emitted (the gate is the whole point).

    The (chromatic) defringe is NOT done here -- the caller runs the existing
    ResUNet defringer afterwards. Returns numpy HWC float32.
    """
    import torch
    import torch.nn.functional as F
    from .deconv import richardson_lucy_fft, wiener_fft, gaussian_psf, edge_gate_from_l1

    sigma = float(cfg.get("sigma", 1.1))
    iters = int(cfg.get("iters", 3))
    method = str(cfg.get("method", "rl"))
    nsr = float(cfg.get("nsr", 1e-2))

    t = torch.as_tensor(np.ascontiguousarray(img), dtype=torch.float32, device="cuda")
    t = t.permute(2, 0, 1).clamp(0.0, 1.0)  # (3, H, W)
    H, W = t.shape[-2:]
    psf = gaussian_psf(sigma)

    out = torch.empty_like(t)
    for c in range(t.shape[0]):
        if method == "rl":
            out[c] = richardson_lucy_fft(t[c], psf, iters=iters, clip=True)
        elif method == "wiener":
            out[c] = wiener_fft(t[c], psf, nsr=nsr).clamp(0.0, 1.0)
        else:
            raise ValueError(f"unknown deconv method {method!r} (use 'rl' or 'wiener')")

    if edge_strength is not None:
        gate = edge_gate_from_l1(
            edge_strength,
            lo_pct=float(cfg.get("gate_lo_pct", 50)),
            hi_pct=float(cfg.get("gate_hi_pct", 90)),
        ).to(t.device)
        if gate.shape != (H, W):  # align the gate to the image (soft mask -> bilinear)
            gate = F.interpolate(gate[None, None], size=(H, W),
                                 mode="bilinear", align_corners=False)[0, 0]
        out = t + gate[None] * (out - t)  # blend only at edges
    else:
        warnings.warn(
            "measured_rl deblur ran without an edge_strength gate (set "
            "merging.expose_edge_strength: true); deconvolving the whole frame, "
            "which amplifies flat-region noise.")

    out = out.clamp(0.0, 1.0)
    result = out.permute(1, 2, 0).contiguous().cpu().numpy()

    # optional edge-preserving bilateral to strip the noise the deconv amplified
    if bool(cfg.get("bilateral", False)):
        import cv2
        sc = cfg.get("bilateral_sigma_color", None)
        if sc is None:
            # ~3x the frame's high-pass std (a noise-floor proxy), as in eval.deconv
            g = result.mean(2).astype(np.float32)
            sc = 3.0 * float((g - cv2.GaussianBlur(g, (0, 0), 2.0)).std())
        sc = max(float(sc), 1e-4)
        result = cv2.bilateralFilter(np.ascontiguousarray(result), d=5,
                                     sigmaColor=sc, sigmaSpace=3)
        result = np.clip(result, 0.0, 1.0).astype(np.float32)
    return result.astype(np.float32)


_ABERRATION_MODEL = None


def _aberration(img, cfg, edge_strength=None):
    """Two-stage optical-aberration correction on linear-RGB HWC in [0, 1]:
    a deblur stage + a learned ResUNet chromatic defringer. Vendored from
    teboli/fast_two_stage_psf_correction (MIT). The model (with its checkpoint) is
    cached across calls, keyed on its build parameters.

    The deblur stage is selected by ``cfg.deblur_method`` (default ``polyblur``):
      * ``polyblur`` -- the shipped per-tile (spatially-varying) blind, bounded
        Polyblur deblur (unchanged behavior);
      * ``measured_rl`` -- an opt-in measured-PSF Richardson-Lucy deconvolution
        (``_measured_rl_deblur``), gated by the merge's structure-tensor
        ``edge_strength`` field. In this mode the deblur is run first on the CPU/GPU
        and the model is invoked for the DEFRINGE only (``do_deblur=False``).
    """
    import torch
    from .aberration import AberrationCorrection

    deblur_method = str(cfg.get("deblur_method", "polyblur"))
    do_deblur = bool(cfg.get("do_deblur", True))
    do_defringe = bool(cfg.get("do_defringe", True))

    if deblur_method == "measured_rl":
        # Measured-PSF RL replaces the Polyblur deblur; the model then defringes.
        if do_deblur:
            img = _measured_rl_deblur(img, cfg, edge_strength=edge_strength)
        model_do_deblur = False
    elif deblur_method == "polyblur":
        model_do_deblur = do_deblur
    else:
        raise ValueError(
            f"unknown deblur_method {deblur_method!r} (use 'polyblur' or 'measured_rl')")

    global _ABERRATION_MODEL
    key = (cfg.get("model_type", "super_tiny"), int(cfg.get("patch_size", 400)),
           float(cfg.get("overlap", 0.25)), int(cfg.get("batch_size", 20)),
           int(cfg.get("ker_size", 31)))
    if _ABERRATION_MODEL is None or _ABERRATION_MODEL[0] != key:
        _ABERRATION_MODEL = (key, AberrationCorrection(
            model_type=key[0], patch_size=key[1], overlap=key[2],
            batch_size=key[3], ker_size=key[4]).cuda())
    model = _ABERRATION_MODEL[1]

    t = torch.as_tensor(np.ascontiguousarray(img), dtype=torch.float32, device="cuda")
    t = t.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)  # (1, 3, H, W)
    out = model(t, c=float(cfg.get("c", 0.416)), sigma_b=float(cfg.get("sigma_b", 0.358)),
                alpha=float(cfg.get("alpha", 3.0)), b=float(cfg.get("beta", 3.0)),
                do_deblur=model_do_deblur,
                do_defringe=do_defringe)
    return out.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()


def postprocess(raw, img=None, do_color_correction=True, do_tonemapping=True,
                do_gamma=True, sharpening_config=None, do_devignette=False, xyz2cam=None,
                deblur_config=None, aberration_config=None, edge_strength=None):
    """
    Convert a raw image to jpg image.

    ``edge_strength`` (optional) is the merge's structure-tensor l1 field (from
    ``debug_dict['edge_strength']`` when ``merging.expose_edge_strength`` is on),
    used only by the opt-in ``aberration.deblur_method == 'measured_rl'`` path to
    gate the deconvolution. Ignored otherwise.
    """
    if img is None:
        ## Rawpy processing - whole stack
        return img_as_float32(raw.postprocess(use_camera_wb=True))
    else:
        ## Color matrix
        if do_color_correction:
            ## First, read the color matrix from rawpy
            rgb2cam = get_color_matrix(raw, xyz2cam)
            cam2rgb = np.linalg.inv(rgb2cam)
            img = apply_ccm(img, cam2rgb)
            img = np.clip(img, 0.0, 1.0)
        ## Aberration (Stage B) > deblur (Stage A) > sharpen -- mutually exclusive
        if aberration_config is not None and aberration_config.enabled:
            img = _aberration(img, aberration_config, edge_strength=edge_strength)
        elif deblur_config is not None and deblur_config.enabled:
            img = _deblur(img, deblur_config)
        elif sharpening_config is not None and sharpening_config.enabled:
            if "radius" in sharpening_config and "amount" in sharpening_config:
                img = filters.unsharp_mask(img, radius=sharpening_config.radius,
                                           amount=sharpening_config.amount,
                                           channel_axis=2, preserve_range=True)
            else:
                warnings.warn('Sharpening config is missing radius or amount parameter, using default values.')
                img = filters.unsharp_mask(img, radius=3,
                                           amount=0.5,
                                           channel_axis=2, preserve_range=True)
        ## Devignette
        if do_devignette:
            img = devignette(img)
        ## Tone mapping
        if do_tonemapping:
            img = apply_smoothstep(img)
        img = np.clip(img, 0.0, 1.0)
        ## Gamma compression
        if do_gamma:
            img = gamma_compression(img)
        img = np.clip(img, 0.0, 1.0)
        return img
