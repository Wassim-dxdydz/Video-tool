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
