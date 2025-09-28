import argparse, time, cv2, numpy as np
from processing import zebra_overlay, safe_frame_bgr

def draw_hud(img, text):
    out = img.copy()
    org = (12, 24)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    return out

def get_screen_size():
    # Works on Windows; safe fallback otherwise
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1280, 720  # fallback

def parse_args():
    p = argparse.ArgumentParser(description="Live zebra preview (auto-fit to screen).")
    p.add_argument("--video", required=True)
    p.add_argument("--mode", choices=["Over","Under","Both"], default="Both")
    p.add_argument("--black", type=int, default=16)
    p.add_argument("--white", type=int, default=235)
    p.add_argument("--scale", type=float, default=0.5, help="Requested scale (0.05..1.0); will be auto-clamped to fit screen")
    p.add_argument("--phase-step", type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()
    req_scale = float(np.clip(args.scale, 0.05, 1.0))
    t0, frames_shown, fps_est = time.time(), 0, 0.0
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("Cannot open video")

    # Make the window resizable
    win = "Zebra Preview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    phase = 0
    paused = False
    mode, black, white = args.mode, int(args.black), int(args.white)
    screen_w, screen_h = get_screen_size()
    max_w, max_h = int(screen_w * 0.9), int(screen_h * 0.9)  # keep some margin

    print("[Keys] q=quit  space=pause/play  m=mode  a/z=black±1  s/x=white±1  [ ]=scale±")

    out = None
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            frame = safe_frame_bgr(frame)

            h0, w0 = frame.shape[:2]

            fit_scale = min(max_w / w0, max_h / h0, 1.0)
            eff_scale = min(req_scale, fit_scale)
            if eff_scale < 1.0:
                frame = cv2.resize(frame, (int(w0 * eff_scale), int(h0 * eff_scale)), interpolation=cv2.INTER_AREA)
                
            out = zebra_overlay(frame, mode, black, white, phase=phase)
            frames_shown += 1
            dt = time.time() - t0
            if dt >= 0.5:
                fps_est = frames_shown / dt
                t0, frames_shown = time.time(), 0
            out = draw_hud(out, f"{mode}  B:{black}  W:{white}  scale:{eff_scale:.2f}  FPS~{fps_est:.1f}")

            phase = (phase + args.phase_step) % 10000

            h, w = out.shape[:2]
            cv2.resizeWindow(win, w, h)

        cv2.imshow(win, out if out is not None else np.zeros((240, 320, 3), np.uint8))

        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            break
        elif k == ord(' '):
            paused = not paused
        elif k == ord('m'):
            mode = {"Over":"Under","Under":"Both","Both":"Over"}[mode]; print("Mode:", mode)
        elif k == ord('a'):
            black = max(0, black-1);  print("Black:", black)
            if black >= white: white = min(255, black+1)
        elif k == ord('z'):
            black = min(254, black+1); print("Black:", black)
            if black >= white: white = min(255, black+1)
        elif k == ord('s'):
            white = max(1, white-1);   print("White:", white)
            if black >= white: black = max(0, white-1)
        elif k == ord('x'):
            white = min(255, white+1); print("White:", white)
            if black >= white: black = max(0, white-1)
        elif k == ord('['):   # smaller
            req_scale = float(np.clip(req_scale - 0.05, 0.05, 1.0))
        elif k == ord(']'):   # larger
            req_scale = float(np.clip(req_scale + 0.05, 0.05, 1.0))


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
