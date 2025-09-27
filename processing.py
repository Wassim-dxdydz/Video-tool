import numpy as np
import cv2

def to_luminance_b709(bgr: np.ndarray) -> np.ndarray:
    """
    Compute BT.709 luma from a BGR frame (uint8 -> float32), range ~0..255.
    Y = 0.2126 R + 0.7152 G + 0.0722 B
    """
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def compute_masks_from_luma(Y: np.ndarray, black: int, white: int):
    """Return boolean masks for under/over given luma array Y."""
    under = (Y <= black)
    over  = (Y >= white)
    return under, over

def frame_stats_from_luma(Y: np.ndarray, black: int, white: int):
    """
    Return (mean luminance, % under black, % over white) for one frame.
    Percentages are 0..100.
    """
    n = Y.size
    under, over = compute_masks_from_luma(Y, black, white)
    mean_y = float(Y.mean())
    pct_under = 100.0 * under.sum() / n
    pct_over  = 100.0 * over.sum()  / n
    return mean_y, pct_under, pct_over

def zebra_overlay(frame_bgr: np.ndarray, mode: str, black: int, white: int,
                  phase: int, period: int = 14, duty: int = 6, alpha: float = 0.55) -> np.ndarray:
    """
    Draw animated diagonal 'zebra' stripes over over/under-threshold areas.
    mode: 'Over' | 'Under' | 'Both'
    """
    h, w = frame_bgr.shape[:2]

    # BT.709 luminance
    b = frame_bgr[..., 0].astype(np.float32)
    g = frame_bgr[..., 1].astype(np.float32)
    r = frame_bgr[..., 2].astype(np.float32)
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b

    over = (Y >= white) if mode in ('Over', 'Both') else None
    under = (Y <= black) if mode in ('Under', 'Both') else None
    if over is None and under is None:
        return frame_bgr

    mask = over if under is None else (under if over is None else np.logical_or(over, under))

    # Diagonal stripe pattern (animated)
    yy, xx = np.indices((h, w))
    pattern = ((xx + yy + phase) % period) < duty  # boolean

    zebra_img = np.where(pattern[..., None], 255, 0).astype(np.uint8)
    zebra_img = np.repeat(zebra_img, 3, axis=2)

    mask3 = np.repeat(mask[..., None], 3, axis=2)
    out = frame_bgr.copy()
    out[mask3] = cv2.addWeighted(out[mask3], 1.0 - alpha, zebra_img[mask3], alpha, 0)
    return out
