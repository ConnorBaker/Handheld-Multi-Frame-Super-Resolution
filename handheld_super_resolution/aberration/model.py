"""PROTOTYPE: two-stage optical-aberration correction (Stage B).

Spatially-varying Polyblur deblur (per overlapping tile) + a learned ResUNet
chromatic defringer, vendored/adapted from teboli/fast_two_stage_psf_correction
(MIT, (c) 2022 Thomas Eboli), ECCV'22.

Key prototype decisions vs. upstream `fast_optics_correction/modules.py`:
  * REUSE our vendored Polyblur for the per-tile blur estimation + polynomial
    inverse (gaussian_blur_estimation + inverse_filtering_rank3) instead of the
    older duplicated `polyblur_functions.py`.
  * REPLACE kornia.contrib.extract_tensor_patches / compute_padding with
    torch.nn.functional.unfold / fold (no kornia dependency).
  * ResUNet + basicblock are copied verbatim (the genuinely-new learned piece).
All local edits to run on torch 2.12 / numpy 2.4 are marked `# vendored-fix`.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import basicblock as B               # vendored-fix: package-relative import
from ..polyblur import deblurring as pb_deblur  # vendored-fix: reuse our Polyblur


# ----------------------------------------------------------------------------
# ResUNet defringer (verbatim from fast_optics_correction/modules.py)
# ----------------------------------------------------------------------------
class ResUNet(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R',
                 downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(ResUNet, self).__init__()
        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError(downsample_mode)
        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))
        self.m_body = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError(upsample_mode)
        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        x = x[..., :h, :w]
        return x


def _build_window(ps, beta=5.0, device='cpu'):
    wi = torch.kaiser_window(ps, beta=beta, periodic=True, device=device)
    wj = torch.kaiser_window(ps, beta=beta, periodic=True, device=device)
    return (wi.unsqueeze(-1) * wj.unsqueeze(0)).view(1, 1, ps, ps)


class AberrationCorrection(nn.Module):
    """Per-tile Polyblur deblur + ResUNet defringe over a kaiser-windowed fold.

    forward(image): image is (1,3,H,W) in [0,1] linear RGB on any device.
    """

    def __init__(self, model_type='super_tiny', patch_size=400, overlap=0.25, batch_size=20,
                 ker_size=31, checkpoint_dir=None, load_weights=True):
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.ker_size = ker_size
        if model_type == 'tiny':
            self.defringer = ResUNet(nc=[16, 32, 64, 64], in_nc=2, out_nc=1)
        elif model_type == 'super_tiny':
            self.defringer = ResUNet(nc=[16, 16, 32, 32], in_nc=2, out_nc=1)
        else:
            self.defringer = ResUNet(nc=[64, 128, 256, 512], in_nc=2, out_nc=1)
        if load_weights:
            # vendored-fix: resolve the checkpoint next to this file (cwd-independent)
            if checkpoint_dir is None:
                checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
            ckpt = os.path.join(checkpoint_dir, model_type + '_epoch_1000.pt')
            self.defringer.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
        self.defringer.eval()

    @torch.no_grad()
    def forward(self, image, c=0.416, sigma_b=0.358, q=0.0, alpha=6.0, b=1.0,
                polyblur_iters=1, do_defringe=True, do_deblur=True):
        assert image.shape[0] == 1
        device = image.device
        ps = self.patch_size
        # Make dims even
        h, w = image.shape[-2:]
        if h % 2: image = image[..., :-1, :]; h -= 1
        if w % 2: image = image[..., :, :-1]; w -= 1

        stride = int(ps * (1 - self.overlap))
        # Pad so an integer number of strided tiles tile the image (replicate).
        # n tiles along a dim of size L: ceil((L-ps)/stride)*stride + ps
        new_h = int(np.ceil(max(h - ps, 0) / stride) * stride) + ps
        new_w = int(np.ceil(max(w - ps, 0) / stride) * stride) + ps
        pad = (0, new_w - w, 0, new_h - h)
        img_p = F.pad(image, pad, mode='replicate')

        window = _build_window(ps, device=device)

        # --- unfold into tiles: F.unfold gives (1, C*ps*ps, L) ---
        patches = F.unfold(img_p, kernel_size=ps, stride=stride)  # (1, 3*ps*ps, L)
        L = patches.shape[-1]
        patches = patches.permute(0, 2, 1).reshape(L, 3, ps, ps)  # (L,3,ps,ps)

        out_patches = torch.empty_like(patches)
        bs = self.batch_size
        for n in range(0, L, bs):
            patch = patches[n:n + bs]
            if do_deblur:
                # Reuse the proven high-level Polyblur entry point per batch of tiles
                # (same call the live raw2rgb._deblur uses, just per-tile). This
                # gives the spatial variation: each tile gets its own estimated
                # blur kernel + polynomial inverse.
                patch = pb_deblur.polyblur_deblurring(
                    patch, n_iter=polyblur_iters, c=c, b=sigma_b, q=q,
                    alpha=alpha, beta=b, ker_size=self.ker_size,
                    remove_halo=False, prefiltering=False, edgetaping=False)
                patch = patch.clamp(0, 1)
            if do_defringe:
                # R defringed using (R,G); B defringed using (B,G). Predict residual.
                r = patch[:, 0:1] - self.defringer(patch[:, 0:2])
                bch = patch[:, 2:3] - self.defringer(torch.cat([patch[:, 2:3], patch[:, 1:2]], dim=1))
                patch = torch.cat([r, patch[:, 1:2], bch], dim=1)
                patch = patch.clamp(0, 1)
            out_patches[n:n + bs] = patch

        # --- fold back with window weighting ---
        weighted = (out_patches * window).reshape(L, 3 * ps * ps).permute(1, 0).unsqueeze(0)
        folded = F.fold(weighted, output_size=(new_h, new_w), kernel_size=ps, stride=stride)
        wcols = window.repeat(L, 1, 1, 1).reshape(L, ps * ps).permute(1, 0).unsqueeze(0)
        wsum = F.fold(wcols, output_size=(new_h, new_w), kernel_size=ps, stride=stride)
        out = folded / (wsum + 1e-8)
        out = out[..., :h, :w].clamp(0, 1)
        return out
