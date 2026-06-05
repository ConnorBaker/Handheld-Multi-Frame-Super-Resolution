import numpy as np
import torch
import torch.fft

from . import edgetaper
from . import filters
from . import blur_estimation
from . import utils

from time import time


#####################################################
################## Functional #######################
#####################################################



def polyblur_deblurring(img, n_iter=1, c=0.352, b=0.768, alpha=2, beta=3, sigma_r=0.8, sigma_s=2.0, ker_size=25, q=0.0, n_angles=6,
                        n_interpolated_angles=30, remove_halo=False, edgetaping=False, prefiltering=False, 
                        discard_saturation=False, multichannel_kernel=False, method='fft', verbose=False):
    """
    Meta Functional implementation of Polyblur.
    :param img: (H, W) or (H,W,3) np.array or (B,C,H,W) torch.tensor, the blurry image(s)
    :param n_iter: int, number of polyblur iteration
    :param c: float, slope of std estimation model (C in the main paper)
    :param b: float, intercept of the std estimation model
    :param alpha: float, mid frequencies parameter for deblurring
    :param beta: float, high frequencies parameter for deblurring
    :param sigma_r: float, regularization parameter for domain transform (prefiltering)
    :param sigma_s: float, smoothness parameter for domain transform (prefiltering)
    :param ker_size: int, kernel(s) size
    :param remove_halo: bool, using or not halo removal masking
    :param edgetaping: bool, using or not edgetaping border preprocessing for deblurring
    :param prefiltering: bool, using or not smooth and textured images split to prevent noise magnification
    :param discard_saturation: bool, should we discard saturated pixels? (recommanded)
    :param multichannel_kernel: bool, predicting a kernel common to each color channel or a single kernel per color channel
    :param method: string, weither running the convolution in time or fourier domain
    :return: impred: np.array or torch.tensor of same size as img, the restored image(s)
    """
    ## Convert to torch.Tensor from numpy.ndarray
    if type(img) == np.ndarray:
        flag_numpy = True
        img = utils.to_tensor(img).unsqueeze(0)
    else:
        flag_numpy = False

    ## Which format for the kernel: 2D filter (FFT or direct) or tuple of parameters (direct separable)?
    if method == 'direct_separable':
        return_2d_filters = False
    else:
        return_2d_filters = True

    ## Init the variables
    start = time()
    impred = img
    grad_img = filters.fourier_gradients(img)
    thetas = torch.linspace(0, 180, n_angles+1, device=img.device).unsqueeze(0).long()   # (1,n)
    interpolated_thetas = torch.arange(0, 180, 180 / n_interpolated_angles, device=img.device).unsqueeze(0).long()  # (1,N)
    if verbose:
        print('-- init tensors:      %1.5f' % (time() - start))

    ## Main loop
    for n in range(n_iter):
        ## Blur estimation
        start = time()
        kernel = blur_estimation.gaussian_blur_estimation(impred, c=c, b=b, q=q, discard_saturation=discard_saturation, 
                                                          ker_size=ker_size, multichannel=multichannel_kernel, 
                                                          thetas=thetas, interpolated_thetas=interpolated_thetas,
                                                          return_2d_filters=return_2d_filters)
        if verbose:
            print('-- blur estimation %d: %1.5f' % (n+1, time() - start))

        ## Non-blind deblurring
        start =time()
        if prefiltering:
            impred, impred_noise = edge_aware_filtering(impred, sigma_s, sigma_r)
            impred = inverse_filtering_rank3(impred, kernel, alpha=alpha, b=beta, remove_halo=remove_halo,
                                             do_edgetaper=edgetaping, grad_img=grad_img)
            impred = impred + impred_noise
        else:
            impred = inverse_filtering_rank3(impred, kernel, alpha=alpha, b=beta, remove_halo=remove_halo,
                                             do_edgetaper=edgetaping, grad_img=grad_img)
        impred = impred.clip(0.0, 1.0)
        if verbose:
            print('-- deblurring %d:      %1.5f' % (n+1, time() - start))

    ## Go back to numpy if needs be
    if flag_numpy:
        return utils.to_array(impred.cpu())
    else:
        return impred


def edge_aware_filtering(img, sigma_s, sigma_r):
    """
    Separate img into smooth and detail (noise) components using the the edge aware smoothing. (Alg.4 and Alg.6)
    :param img: (B,C,H,W) torch.tensor, the input image(s)
    :param sigma_r: float, regularization parameter for domain transform
    :param sigma_s: float, smoothness parameter for domain transform
    :return: img_smoothed, img_noise: torch.tensors of same size as img, the smooth and noise components of img
    """
    # img_smoothed = domain_transform.recursive_filter(img, sigma_r=sigma_r, sigma_s=sigma_s, num_iterations=1)
    img_smoothed = filters.bilateral_filter(img)
    img_noise = img - img_smoothed
    return img_smoothed, img_noise


def compute_polynomial_fft(img, kernel, alpha, b, not_symmetric=False):
    """
    Implements in the fourier domain the polynomial deconvolution filter (Deconvolution Alg.4) 
    using the polynomial approximation of Eq. (27)
    :param Y: (B,C,H,W) torch.tensor, the blurry image(s)
    :param K: (B,C,h,w)  or (B,1,h,w) torch.tensor, the blur kernel(s)
    :param alpha: float, mid frequencies parameter for deblurring
    :param beta: float, high frequencies parameter for deblurring
    :return torch.tensor of same size as img, the deblurred image(s)
    """
    ## Go to Fourier domain
    h, w = img.shape[-2:]
    Y = torch.fft.fft2(img, dim=(-2, -1))
    K = filters.p2o(kernel, (h, w))  # from NxCxhxw to NxCxHxW
    ## If the filter is not symmetric, shift the image
    if not_symmetric:
        C = torch.conj(K) / (torch.abs(K) + 1e-8)
        Y = C * Y
    ## C implements the pure phase filter described in the Polyblur article 
    ## needed to deblur non symmetic kernels. For Gaussian kernels (symmetric) it has no effect.
    a3 = alpha / 2 - b + 2
    a2 = 3 * b - alpha - 6
    a1 = 5 - 3 * b + alpha / 2
    X = a3 * Y
    X = K * X + a2 * Y
    X = K * X + a1 * Y
    X = K * X + b * Y
    ## Go back to temporal domain
    return torch.real(torch.fft.ifft2(X, dim=(-2, -1)))


# @torch.jit.script
def grad_prod_(grad_x, grad_y, gout_x, gout_y):
    return (- grad_x * gout_x) +  (- grad_y * grad_y)


# @torch.jit.script
def grad_square_(grad_x, grad_y):
    return grad_x * grad_x + grad_y * grad_y


# @torch.jit.script
def grad_div_and_clip_(M, nM):
    return torch.clamp(M / (nM + M), min=0)


# @torch.jit.script
def grad_convex_sum_(img, imout, z):
    # Equivalent to z * img + (1-z) * imout
    return imout + z * (img - imout)


def halo_masking(img, imout, grad_img=None):
    """
    Halo removal processing. Detects gradient inversions between input and deblurred image replaces them in the output (Alg.5)
    :param img: (B,C,H,W) torch.tensor, the blurry image(s)
    :param imout: (B,C,H,W) torch.tensor, the deblurred image(s)
    :return torch.tensor of same size as img, the halo corrected image(s)
    """
    if grad_img is None:
        grad_x, grad_y = filters.fourier_gradients(img)
    else:
        grad_x, grad_y = grad_img
    gout_x, gout_y = filters.fourier_gradients(imout)
    M = grad_prod_(grad_x, grad_y, gout_x, gout_y)
    nM = torch.sum(grad_square_(grad_x, grad_y), dim=(-2, -1), keepdim=True)
    z = grad_div_and_clip_(M, nM)
    return grad_convex_sum_(img, imout, z)


def inverse_filtering_rank3(img, kernel, alpha=2, b=4, correlate=False, remove_halo=False, 
                            do_edgetaper=False, grad_img=None):
    """
    Deconvolution with approximate inverse filter parameterized by alpha and beta. (Deconvolution Alg.4, EdgeAwareFiltering is in Alg.1)
    :param img: (B,C,H,W) torch.tensor, the blurry image(s)
    :param kernel: (B,C,h,w) torch.tensor, the blur kernel(s)
    :param alpha: float, mid frequencies parameter
    :param beta: float, high frequencies parameter
    :param correlate: bool, deconvolving with a correlation or not
    :param masking: bool, using or not halo removal masking
    :param do_edgetaper bool, using or not edgetaping border preprocessing for deblurring
    :return torch.tensor of same size as img, the deblurred image(s)
    """
    if correlate:
        kernel = torch.rot90(kernel, k=2, dims=(-2, -1))
    ## Pad (with edgetaping optionally)
    img = utils.pad_with_kernel(img, kernel)
    if do_edgetaper:
        img = edgetaper.edgetaper(img, kernel)  # for better edge handling
    ## Deblurring
    imout = compute_polynomial_fft(img, kernel, alpha, b)
    ## Crop
    imout = utils.crop_with_kernel(imout, kernel)
    ## Mask deblurring halos
    if remove_halo:
        img = utils.crop_with_kernel(img, kernel)
        imout = halo_masking(img, imout, grad_img)
    return torch.clamp(imout, 0.0, 1.0)
