import argparse, cv2
from processing import zebra_overlay

p = argparse.ArgumentParser(); 
p.add_argument("--video", required=True); p.add_argument("--out", default="zebra_preview.mp4")
p.add_argument("--mode", choices=["Over","Under","Both"], default="Both")
p.add_argument("--black", type=int, default=16); p.add_argument("--white", type=int, default=235)
p.add_argument("--frames", type=int, default=200)
args = p.parse_args()

cap = cv2.VideoCapture(args.video); 
if not cap.isOpened(): raise SystemExit("Cannot open video")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
wr = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
phase = 0
for i in range(args.frames):
    ok, frame = cap.read()
    if not ok: break
    out = zebra_overlay(frame, args.mode, args.black, args.white, phase=phase)
    phase = (phase + 2) % 10000
    wr.write(out)
cap.release(); wr.release()
print("Wrote", args.out)
