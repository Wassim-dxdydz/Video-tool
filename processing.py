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
    # Normalize input
    frame_bgr = safe_frame_bgr(frame_bgr)
    h, w = frame_bgr.shape[:2]

    # Clamp thresholds and enforce black < white
    black = int(np.clip(black, 0, 254))
    white = int(np.clip(white, 1, 255))
    if black >= white:
        white = min(255, black + 1)

    # BT.709 luminance (0..255)
    b = frame_bgr[..., 0].astype(np.float32)
    g = frame_bgr[..., 1].astype(np.float32)
    r = frame_bgr[..., 2].astype(np.float32)
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Build mask(s)
    over  = (Y >= white) if mode in ('Over', 'Both') else None
    under = (Y <= black) if mode in ('Under', 'Both') else None
    if over is None and under is None:
        return frame_bgr
    mask = over if under is None else (under if over is None else np.logical_or(over, under))  # (H,W) bool

    # Diagonal stripe pattern (animated)
    yy, xx = np.indices((h, w))
    pattern = ((xx + yy + phase) % period) < duty  # (H,W) bool

    # 3-channel zebra image
    zebra_1ch = np.where(pattern, 255, 0).astype(np.uint8)         # (H,W)
    zebra_img = np.repeat(zebra_1ch[:, :, None], 3, axis=2)        # (H,W,3)

    # Blend once, then select via mask (broadcasts cleanly)
    blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, zebra_img, alpha, 0)  # (H,W,3)
    out = np.where(mask[:, :, None], blended, frame_bgr)
    return out

def safe_frame_bgr(frame):
    """Ensure uint8 3-channel BGR (convert from Gray/BGRA/float if needed)."""
    if frame is None:
        return None
    # Convert dtype if needed (e.g., float frames)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    # Handle Gray and BGRA
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame

def resize_keep_ar(frame_bgr, max_w: int, max_h: int):
    """Downscale to fit within (max_w, max_h) keeping aspect ratio (no upscaling)."""
    h, w = frame_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame_bgr
