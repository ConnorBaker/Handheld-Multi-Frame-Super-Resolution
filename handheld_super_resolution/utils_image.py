import math

import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndx
from scipy.ndimage._filters import _gaussian_kernel1d
from numba import cuda
import torch as th
import torch.fft
import torch.nn.functional as F

from .utils import getSigned, DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_TORCH_FLOAT_TYPE, DEFAULT_THREADS

def apply_orientation(img, ori):
    """
    Applies an orientation to an image

    Parameters
    ----------
    img : numpy Array [ny, nx, c]
        Image
    ori : int
        Exif orientation as defined here:
            https://exiftool.org/TagNames/EXIF.html

    Returns
    -------
    Oriented image

    """
    
    if ori == 1:
        pass
    elif ori == 2:
        # Mirrored horizontal
        img = np.flip(img, axis=1)
    elif ori == 3:
        # Rotate 180
        img = np.rot90(img, k=2, axes=(0, 1))
    elif ori == 4:
        # Mirror vertical
        img = np.flip(img, axis=0)
    elif ori == 5:
        # Mirror horizontal and rotate 270 CW
        img = np.flip(img, axis=1)
        img = np.rot90(img, k=-3, axes=(0, 1))
    elif ori == 6:
        # Rotate 90 CW
        img = np.rot90(img, k=-1, axes=(0, 1))
    elif ori == 7:
        # Mirror horizontal and rotate 90 CW
        img = np.flip(img, axis=1)
        img = np.rot90(img, k=-1, axes=(0, 1))
    elif ori == 8:
        # Rotate 270 CW
        img = np.rot90(img, k=-3, axes=(0, 1))
    
    return img

def compute_grey_images(img, method):
    """
    This function converts a raw image to a grey image, using the decimation or
    the method of Alg. 3: ComputeGrayscaleImage

    Parameters
    ----------
    img : device Array[:, :]
        Raw image J to convert to gray level.
    method : str
        FFT or decimatin.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    img_grey : device Array[:, :]
        Corresponding grey scale image G

    """
    imsize_y, imsize_x = img.shape
    if method == "FFT":
        torch_img_grey = th.as_tensor(img, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
        torch_img_grey = torch.fft.fft2(torch_img_grey) 
        # th FFT induces copy on the fly : this is good because we dont want to 
        # modify the raw image, it is needed in the future
        # Note : the complex dtype of the fft2 is inherited from DEFAULT_TORCH_FLOAT_TYPE.
        # Therefore, for DEFAULT_TORCH_FLOAT_TYPE = float32 we directly get complex64
        torch_img_grey = torch.fft.fftshift(torch_img_grey)
        
        torch_img_grey[:imsize_y//4, :] = 0
        torch_img_grey[:, :imsize_x//4] = 0
        torch_img_grey[-imsize_y//4:, :] = 0
        torch_img_grey[:, -imsize_x//4:] = 0
        
        torch_img_grey = torch.fft.ifftshift(torch_img_grey)
        torch_img_grey = torch.fft.ifft2(torch_img_grey)
        # Here, .real() type inherits once again from the complex type.
        # numba type is read directly from the torch tensor, so everything goes fine.
        return cuda.as_cuda_array(torch_img_grey.real)
    elif method == "decimating":
        # CuPy 2x2 block average (numba cuda_decimate_to_grey can't JIT here)
        return compute_grey_images_decimate_cupy(img)

    else:
        raise NotImplementedError('Computation of gray level on GPU is only supported for FFT')


# ---------------------------------------------------------------------------
# CuPy implementations
#
# Drop-in equivalents of the torch / numba routines above, ported from the
# feat/warp branch. They exist so the pipeline can run on GPUs where numba's
# CUDA target cannot JIT (e.g. when ptxas rejects the PTX version numba emits).
# Each is covered by an equivalence test in test/test_utils_image.py.
# ---------------------------------------------------------------------------

def compute_grey_images_fft_cupy(img):
    """CuPy equivalent of ``compute_grey_images(img, "FFT")``.

    Returns a ``cp.ndarray`` (the torch path returns a numba device array).
    """
    imsize_y, imsize_x = img.shape
    cp_img_grey = cp.asarray(img, dtype=cp.float32)
    cp_img_grey = cp.fft.fft2(cp_img_grey)
    cp_img_grey = cp.fft.fftshift(cp_img_grey)

    cp_img_grey[:imsize_y // 4, :] = 0
    cp_img_grey[:, :imsize_x // 4] = 0
    cp_img_grey[-imsize_y // 4:, :] = 0
    cp_img_grey[:, -imsize_x // 4:] = 0

    cp_img_grey = cp.fft.ifftshift(cp_img_grey)
    cp_img_grey = cp.fft.ifft2(cp_img_grey)
    return cp_img_grey.real


def compute_grey_images_decimate_cupy(img):
    """CuPy equivalent of ``compute_grey_images(img, "decimating")``.

    Averages 2x2 blocks. The leading odd row/column (if any) is dropped, to
    match the ``imsize // 2`` output shape of the numba ``cuda_decimate_to_grey``.
    """
    img = cp.asarray(img)
    h, w = img.shape
    img = img[:2 * (h // 2), :2 * (w // 2)]
    return (img[0::2, 0::2] + img[1::2, 0::2] + img[0::2, 1::2] + img[1::2, 1::2]) / 4


    
    

def frame_count_denoising_gauss(image, r_acc, config):
    denoised = cuda.device_array(image.shape, DEFAULT_NUMPY_FLOAT_TYPE)
    
    grey_mode = config.mode == 'grey'
    scale = config.scale
    sigma_max = config.sigma_max
    max_frame_count = config.max_frame_count

    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS, 1)
    blockspergrid_x = math.ceil(denoised.shape[1]/threadsperblock[1])
    blockspergrid_y = math.ceil(denoised.shape[0]/threadsperblock[0])
    blockspergrid_z = math.ceil(denoised.shape[2]/threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    
    cuda_frame_count_denoising_gauss[blockspergrid, threadsperblock](
        image, denoised, r_acc,
        scale, sigma_max, max_frame_count, grey_mode)
    
    return denoised

@cuda.jit
def cuda_frame_count_denoising_gauss(noisy, denoised, r_acc,
                               scale, sigma_max, max_frame_count, grey_mode):
    x, y, c = cuda.grid(3)
    imshape_y, imshape_x, _ = noisy.shape
    
    if not (0 <= y < imshape_y and
            0 <= x < imshape_x):
        return
    
    if grey_mode:
        y_grey = int(round(y/scale))
        x_grey = int(round(x/scale))
    else:
        y_grey = int(round((y-0.5)/(2*scale)))
        x_grey = int(round((x-0.5)/(2*scale)))
        
    r = r_acc[y_grey, x_grey]
    sigma = denoise_power_gauss(r, sigma_max, max_frame_count)
    
    t = 3*sigma
    
    num = 0
    den = 0
    for i in range(-t, t+1):
        for j in range(-t, t+1):
            x_ = x + j
            y_ = y + i
            if (0 <= y_ < imshape_y and
                0 <= x_ < imshape_x):
                if sigma == 0:
                    w = (i==j==0)
                else:
                    w = math.exp(-(j*j + i*i)/(2*sigma*sigma))
                num += w * noisy[y_, x_, c]
                den += w
                
    denoised[y, x, c] = num/den

@cuda.jit(device=True)
def denoise_power_gauss(r_acc, sigma_max, r_max):
    r = min(r_acc, r_max)
    return sigma_max * (r_max - r)/r_max

def frame_count_denoising_median(image, r_acc, config):
    denoised = cuda.device_array(image.shape, DEFAULT_NUMPY_FLOAT_TYPE)

    grey_mode = config.mode == 'grey'
    scale = config.scale
    radius_max = config.radius_max
    max_frame_count = config.max_frame_count

    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS, 1)
    blockspergrid_x = math.ceil(denoised.shape[1]/threadsperblock[1])
    blockspergrid_y = math.ceil(denoised.shape[0]/threadsperblock[0])
    blockspergrid_z = math.ceil(denoised.shape[2]/threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    
    cuda_frame_count_denoising_median[blockspergrid, threadsperblock](
        image, denoised, r_acc,
        scale, radius_max, max_frame_count, grey_mode)
    
    return denoised

@cuda.jit
def cuda_frame_count_denoising_median(noisy, denoised, r_acc,
                               scale, radius_max, max_frame_count, grey_mode):
    x, y, c = cuda.grid(3)
    imshape_y, imshape_x, _ = noisy.shape
    
    if not (0 <= y < imshape_y and
            0 <= x < imshape_x):
        return
    
    if grey_mode:
        y_grey = int(round(y/scale))
        x_grey = int(round(x/scale))
    else:
        y_grey = int(round((y-0.5)/(2*scale)))
        x_grey = int(round((x-0.5)/(2*scale)))
        
    r = r_acc[y_grey, x_grey]
    radius = denoise_power_median(r, radius_max, max_frame_count)
    radius = min(14, radius) # for memory purpose
    
    
    buffer = cuda.local.array(16*16, DEFAULT_CUDA_FLOAT_TYPE)
    k = 0
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            x_ = x + j
            y_ = y + i
            if (0 <= y_ < imshape_y and
                0 <= x_ < imshape_x):
                buffer[k] = noisy[y_, x_, c]
                k += 1
    
    bubble_sort(buffer[:k])
    
    
    denoised[y, x, c] = buffer[k//2]
    

@cuda.jit(device=True)
def denoise_power_median(r_acc, radius_max, max_frame_count):
    r = min(r_acc, max_frame_count)
    return round(radius_max * (max_frame_count - r)/max_frame_count)

@cuda.jit(device=True)
def bubble_sort(X):
    N = X.size

    for i in range(N-1):
        for j in range(N-i-1):
            if X[j] > X[j+1]:
                X[j], X[j+1] = X[j+1], X[j]
                
    
    

            
                
    

        

def cuda_downsample(th_img, kernel='gaussian', factor=2):
    '''Apply a convolution by a kernel if required, then downsample an image.
    Args:
     	image: Device Array the input image (WARNING: single channel only!)
     	kernel: None / str ('gaussian' / 'bayer') / 2d numpy array
     	factor: downsampling factor
    '''
    # Special case
    if factor == 1:
        return th_img
    
    if kernel is None:
        raise ValueError('use Kernel')
    elif kernel == 'gaussian':
        # gaussian kernel std is proportional to downsampling factor
        # filteredImage = gaussian_filter(image, sigma=factor * 0.5, order=0, output=None, mode='reflect')
        
        # This is the default kernel of scipy gaussian_filter1d
        # Note that pytorch Convolve is actually a correlation, hence the ::-1 flip.
        # copy to avoid negative stride
        gaussian_kernel = _gaussian_kernel1d(sigma=factor * 0.5, order=0, radius=int(4*factor * 0.5 + 0.5))[::-1].copy()
        th_gaussian_kernel = torch.as_tensor(gaussian_kernel, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")

        temp = F.conv2d(th_img, th_gaussian_kernel[None, None, :, None]) # convolve y
        th_filteredImage = F.conv2d(temp, th_gaussian_kernel[None, None, None, :]) # convolve x
    else:
        raise ValueError("please use gaussian kernel")

    # Shape of the downsampled image
    h2, w2 = np.floor(np.array(th_filteredImage.shape[2:]) / float(factor)).astype(int)

    return th_filteredImage[:, :, :h2 * factor:factor, :w2 * factor:factor]


def cupy_downsample(img, kernel='gaussian', factor=2):
    """CuPy equivalent of ``cuda_downsample`` (torch).

    ``img`` is expected to be 4D ``[1, 1, H, W]`` like the torch path. A
    same-size reflect Gaussian filter is applied, then the interior is cropped
    with a ``2 * factor`` offset to reproduce the valid-convolution crop the
    torch version gets from ``F.conv2d`` without padding.
    """
    # Special case
    if factor == 1:
        return img

    if kernel is None:
        raise ValueError('use Kernel')
    elif kernel == 'gaussian':
        # gaussian kernel std is proportional to downsampling factor
        filteredImage = ndx.gaussian_filter(
            cp.asarray(img), sigma=factor * 0.5, order=0, mode='reflect')
    else:
        raise ValueError("please use gaussian kernel")

    # Shape of the downsampled image
    h2, w2 = np.floor(np.array(filteredImage.shape[2:]) / float(factor)).astype(int)

    return filteredImage[
        :, :,
        2 * factor:(h2 - 2) * factor:factor,
        2 * factor:(w2 - 2) * factor:factor]


def compute_gradient_cupy(image):
    """CuPy equivalent of the torch Sobel gradients used in ``ICA.init_ica``.

    Matches ``F.conv2d(image, [-1, 0, 1], padding='same')`` (cross-correlation
    with zero padding) for both axes. Returns ``(gradx, grady)``.
    """
    cp_img = cp.asarray(image, dtype=cp.float32)
    weights = cp.array([-1, 0, 1], dtype=cp.float32)
    gradx = ndx.correlate1d(cp_img, weights, axis=1, mode='constant', cval=0)
    grady = ndx.correlate1d(cp_img, weights, axis=0, mode='constant', cval=0)
    return gradx, grady


def compute_hessian_cupy(gradx, grady, tile_size):
    """CuPy equivalent of the numba ``ICA.compute_hessian`` kernel.

    Uses floor patch counts (``imsize // tile_size``) and drops the remainder
    row/column of pixels, matching the numba kernel and the ``unfold`` tiling
    in ``alignment.init_alignment``. Returns ``(n_patch_y, n_patch_x, 2, 2)``.
    """
    gradx = cp.asarray(gradx)
    grady = cp.asarray(grady)
    H, W = gradx.shape

    n_patch_y = H // tile_size
    n_patch_x = W // tile_size

    gx = gradx[:n_patch_y * tile_size, :n_patch_x * tile_size]
    gy = grady[:n_patch_y * tile_size, :n_patch_x * tile_size]
    gx = gx.reshape(n_patch_y, tile_size, n_patch_x, tile_size)
    gy = gy.reshape(n_patch_y, tile_size, n_patch_x, tile_size)

    gx2 = (gx * gx).sum(axis=(1, 3))
    gy2 = (gy * gy).sum(axis=(1, 3))
    cross = (gx * gy).sum(axis=(1, 3))

    hessian = cp.empty((n_patch_y, n_patch_x, 2, 2), dtype=DEFAULT_NUMPY_FLOAT_TYPE)
    hessian[..., 0, 0] = gx2
    hessian[..., 0, 1] = cross
    hessian[..., 1, 0] = cross
    hessian[..., 1, 1] = gy2
    return hessian


def computeRMSE(image1, image2):
    '''computes the Root Mean Square Error between two images'''
    assert np.array_equal(image1.shape, image2.shape), 'images have different sizes'
    h, w = image1.shape[:2]
    c = 1
    if len(image1.shape) == 3:  # multi-channel image
        c = image1.shape[-1]
    error = getSigned(image1.reshape(h * w * c)) - getSigned(image2.reshape(h * w * c))
    return np.sqrt(np.mean(np.multiply(error, error)))


def computePSNR(image, noisyImage):
    '''computes the Peak Signal-to-Noise Ratio between a "clean" and a "noisy" image'''
    if np.array_equal(image.shape, noisyImage.shape):
        assert image.dtype == noisyImage.dtype, 'images have different data types'
        if np.issubdtype(image.dtype, np.unsignedinteger):
            maxValue = np.iinfo(image.dtype).max
        else:
            assert(np.issubdtype(image.dtype, np.floating) and np.min(image) >= 0. and np.max(image) <= 1.), 'not a float image between 0 and 1'
            maxValue = 1.
        h, w = image.shape[:2]
        c = 1
        if len(image.shape) == 3:  # multi-channel image
            c = image.shape[-1]
        error = np.abs(getSigned(image.reshape(h * w * c)) - getSigned(noisyImage.reshape(h * w * c)))
        mse = np.mean(np.multiply(error, error))
        return 10 * np.log10(maxValue**2 / mse)
    else:
        print('WARNING: images have different sizes: {}, {}. Returning None'.format(image.shape, noisyImage.shape))
        return None
