"""Register the Canon ground-truth image to a pipeline output.

Canon and Samsung differ in sensor, lens, FOV and viewpoint, so we estimate a
homography (SIFT + RANSAC) at a downscaled working resolution and rescale it to
full resolution. A homography only perfectly models a planar scene / pure
rotation, so foreground / large-parallax regions register approximately -- use
DISTS (alignment-robust) for scoring and treat per-tile results accordingly.

Default direction: warp the pipeline output INTO the Canon frame (so we compare
against Canon at its native detail level; a softer config matches worse).
"""

import numpy as np
import cv2


def _gray_u8(rgb):
    return cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def estimate_homography(src_rgb, dst_rgb, work=1600, ratio=0.75, min_inliers=30):
    """Homography H mapping src pixels -> dst pixels, at full resolution.

    Returns (H, info dict). Raises RuntimeError if too few inliers.
    """
    sg, dg = _gray_u8(src_rgb), _gray_u8(dst_rgb)
    sh, sw = sg.shape
    dh, dw = dg.shape
    ss = work / max(sh, sw)
    ds = work / max(dh, dw)
    sgw = cv2.resize(sg, (round(sw * ss), round(sh * ss)))
    dgw = cv2.resize(dg, (round(dw * ds), round(dh * ds)))

    detector = cv2.SIFT_create(nfeatures=6000)
    k1, d1 = detector.detectAndCompute(sgw, None)
    k2, d2 = detector.detectAndCompute(dgw, None)
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    if len(good) < min_inliers:
        raise RuntimeError(f"too few matches: {len(good)}")
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    H_work, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 4.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if H_work is None or inliers < min_inliers:
        raise RuntimeError(f"homography failed: {inliers} inliers")
    # src_full --Ss--> src_work --H_work--> dst_work --Sd^-1--> dst_full
    Ss = np.diag([ss, ss, 1.0])
    Sd = np.diag([ds, ds, 1.0])
    H_full = np.linalg.inv(Sd) @ H_work @ Ss
    return H_full, {"matches": len(good), "inliers": inliers,
                    "inlier_frac": inliers / max(len(good), 1)}


def warp_into(src_rgb, dst_shape, H):
    """Warp src into the dst frame; return (warped_rgb, valid_mask)."""
    dh, dw = dst_shape[:2]
    warped = cv2.warpPerspective(src_rgb.astype(np.float32), H, (dw, dh),
                                 flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.ones(src_rgb.shape[:2], np.float32), H, (dw, dh)) > 0.999
    return warped, valid


def crop_overlap(a, b, valid, pad=8):
    """Crop a, b, valid to the bounding box of the valid mask (minus a margin)."""
    ys, xs = np.where(valid)
    y0, y1 = ys.min() + pad, ys.max() - pad
    x0, x1 = xs.min() + pad, xs.max() - pad
    return a[y0:y1, x0:x1], b[y0:y1, x0:x1], valid[y0:y1, x0:x1]


def color_match(src, ref, mask=None):
    """Match src per-channel mean/std to ref (over mask). Robust to camera color."""
    out = np.empty_like(src)
    for c in range(3):
        s = src[..., c][mask] if mask is not None else src[..., c]
        r = ref[..., c][mask] if mask is not None else ref[..., c]
        out[..., c] = (src[..., c] - s.mean()) / (s.std() + 1e-6) * r.std() + r.mean()
    return np.clip(out, 0, 1)


def register(mine_rgb, canon_rgb, work=1600):
    """Register `mine` into the Canon frame at Canon resolution.

    Returns dict with aligned 'mine', 'canon' (both [0,1], same shape), 'valid'
    mask, and registration 'info'. Color-matched so absolute color is neutralized.
    """
    H, info = estimate_homography(mine_rgb, canon_rgb, work=work)
    warped, valid = warp_into(mine_rgb, canon_rgb.shape, H)
    mine_c, canon_c, valid_c = crop_overlap(warped, canon_rgb, valid)
    mine_c = color_match(mine_c, canon_c, valid_c)
    return {"mine": mine_c, "canon": canon_c, "valid": valid_c, "H": H, "info": info}
