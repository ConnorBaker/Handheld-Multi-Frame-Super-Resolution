import torch
import numpy as np
import torch.fft

from . import utils


#####################################################################
######################## Convolution 2D #############################
#####################################################################

def convolve2d(img, kernel):
    """FFT convolution of (B,C,H,W) images by a (B,C,h,w)/(B,1,h,w) 2D kernel.

    Circular-pads by the kernel half-width, multiplies by the kernel's OTF in the
    Fourier domain, then crops back. (Upstream also had a spatial 'direct' path
    via F.conv2d selected by a `method` arg; the deblur pipeline only ever uses
    this FFT path, so the direct path was removed.)
    """
    X = torch.fft.fft2(utils.pad_with_kernel(img, kernel, mode='circular'))
    K = p2o(kernel, X.shape[-2:])
    return utils.crop_with_kernel(torch.real(torch.fft.ifft2(K * X)), kernel)



#####################################################################
####################### Bilateral filter ############################
#####################################################################


def bilateral_filter(I, ksize=5, sigma_spatial=5.0, sigma_color=0.1):
    ## precompute the spatial kernel: each entry of gw is a square spatial difference
    t = torch.arange(-ksize//2+1, ksize//2+1, device=I.device)
    xx, yy = torch.meshgrid(t, t, indexing='xy')
    gw = torch.exp(-(xx * xx + yy * yy) / (2 * sigma_spatial * sigma_spatial))  # (ksize, ksize)

    ## Create the padded array for computing the color shifts
    I_padded = utils.pad_with_kernel(I, ksize=ksize)

    ## Filtering
    var2_color = 2 * sigma_color * sigma_color
    return bilateral_filter_loop_(I, I_padded, gw, var2_color)


def bilateral_filter_loop_(I, I_padded, gw, var2, do_for=True):
    b, c, h, w = I.shape

    if do_for:  # memory-friendly option (Recommanded for larger images)
        J = torch.zeros_like(I)
        W = torch.zeros_like(I)
        for y in range(gw.shape[0]):
            yy = y + h
            # get the shifted image
            I_shifted = I_padded[..., y:yy, :]
            I_shifted = utils.extract_tiles(I_shifted, kernel_size=(h,w), stride=1)  # (B,ksize,C,H,W)
            # color weight
            F = I_shifted - I.unsqueeze(1)  # (B,ksize,C,H,W)
            F = torch.exp(-F * F / var2) 
            # product with spatial weight
            F *= gw[y].view(-1, 1, 1, 1) # (B,ksize,C,H,W)
            J += torch.sum(F * I_shifted, dim=1)
            W += torch.sum(F, dim=1)
    else:  # pytorch-friendly option (Faster for smaller images and/or batch sizes)
        # get shifted images
        I_shifted = utils.extract_tiles(I_padded, kernel_size=(h,w), stride=1)  # (B,ksize*ksize,C,H,W)
        F = I_shifted - I.unsqueeze(1)
        F = torch.exp( - F * F / var2)  # (B,ksize*ksize,C,H,W)
        # product with spatial weights
        F *= gw.view(-1, 1, 1, 1)
        J = torch.sum(F * I_shifted, dim=1)  # (B,C,H,W)
        W = torch.sum(F, dim=1)  # (B,C,H,W)
    return J / (W + 1e-5)

    


#####################################################################
###################### Classical filters ############################
#####################################################################


# @torch.jit.script
def fourier_gradients(images):
    """
    Compute the image gradients using Fourier interpolation as in Eq. (21a) and (21b)
        :param images: (B,C,H,W) torch.tensor
        :return grad_x, grad_y: tuple of 2 images of same dimensions as images that
                                are the vertical and horizontal gradients
    """
    ## Find fast size for FFT
    h, w = images.shape[-2:]
    h_fast, w_fast = images.shape[-2:]
    # h_fast = scipy.fft.next_fast_len(h)
    # w_fast = scipy.fft.next_fast_len(w)
    ## compute FT
    U = torch.fft.fft2(images, s=(h_fast, w_fast))
    U = torch.fft.fftshift(U, dim=(-2, -1))
    ## Create the freqs components
    freqh = (torch.arange(0, h_fast, device=images.device) - h_fast // 2)[None, None, :, None] / h_fast
    freqw = (torch.arange(0, w_fast, device=images.device) - w_fast // 2)[None, None, None, :] / w_fast
    ## Compute gradients in Fourier domain
    gxU = 2 * np.pi * freqw * (-torch.imag(U) + 1j * torch.real(U))
    gxU = torch.fft.ifftshift(gxU, dim=(-2, -1))
    gxu = torch.real(torch.fft.ifft2(gxU))
    # gxu = crop(gxu, (h, w))
    gyU = 2 * np.pi * freqh * (-torch.imag(U) + 1j * torch.real(U))
    gyU = torch.fft.ifftshift(gyU, dim=(-2, -1))
    gyu = torch.real(torch.fft.ifft2(gyU))
    # gyu = crop(gyu, (h, w))
    return gxu, gyu


#####################################################################
####################### Fourier kernel ##############################
#####################################################################


### From here, taken from https://github.com/cszn/USRNet/blob/master/utils/utils_deblur.py
def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        # otf: NxCxHxWx2
        otf: NxCxHxW
    '''
    otf = torch.zeros(psf.shape[:-2] + shape, dtype=psf.dtype, device=psf.device)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    kh, kw = psf.shape[-2:]
    otf = torch.roll(otf, shifts=(-(kh // 2), -(kw // 2)), dims=(-2, -1))
    otf = torch.fft.fft2(otf, dim=(-2, -1))
    return otf
