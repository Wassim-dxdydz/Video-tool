import argparse, cv2, numpy as np
from processing import zebra_overlay

def parse_args():
    p = argparse.ArgumentParser(description="Live zebra preview (OpenCV window).")
    p.add_argument("--video", required=True)
    p.add_argument("--mode", choices=["Over","Under","Both"], default="Both")
    p.add_argument("--black", type=int, default=16)
    p.add_argument("--white", type=int, default=235)
    p.add_argument("--scale", type=float, default=0.5, help="Preview scale (0.25..1.0)")
    p.add_argument("--phase-step", type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("Cannot open video")

    phase = 0
    paused = False
    mode, black, white = args.mode, args.black, args.white

    print("[Keys] q=quit  space=pause/play  m=mode  a/z=black±1  s/x=white±1")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue  # loop
            if 0 < args.scale < 1.0:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (int(w*args.scale), int(h*args.scale)), interpolation=cv2.INTER_AREA)
            out = zebra_overlay(frame, mode, black, white, phase=phase)
            phase = (phase + args.phase_step) % 10000
        cv2.imshow("Zebra Preview", out)

        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            break
        elif k == ord(' '):
            paused = not paused
        elif k == ord('m'):
            mode = {"Over":"Under","Under":"Both","Both":"Over"}[mode]
            print("Mode:", mode)
        elif k == ord('a'):
            black = max(0, black-1); print("Black:", black)
        elif k == ord('z'):
            black = min(127, black+1); print("Black:", black)
        elif k == ord('s'):
            white = max(128, white-1); print("White:", white)
        elif k == ord('x'):
            white = min(255, white+1); print("White:", white)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()