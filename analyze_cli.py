import argparse, os, sys, csv
import cv2
from processing import to_luminance_b709, frame_stats_from_luma

def parse_args():
    p = argparse.ArgumentParser(description="Analyze first N frames for luminance & clipping.")
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--black", type=int, default=16, help="Black point (default 16)")
    p.add_argument("--white", type=int, default=235, help="White point (default 235)")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames; >0 = cap at N")
    p.add_argument("--top", type=int, default=25, help="How many worst frames to print (0 = all)")
    p.add_argument("--metric", choices=["combined", "over", "under"], default="combined", help="Which metric to rank by")
    p.add_argument("--min-clip", type=float, default=0.0, help="Only show frames with metric >= this percent")
    p.add_argument("--csv", help="Optional: path to write CSV stats")
    return p.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.video):
        print(f"[ERR] File not found: {args.video}")
        sys.exit(2)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("[ERR] Cannot open video. Try an H.264 MP4 file.")
        sys.exit(1)

    fps   = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"# Video: {args.video}")
    print(f"# Size: {w}x{h} | FPS: {fps:.3f} | Frames: {count}")
    print(f"# Thresholds: black={args.black}, white={args.white}")
    print("frame_idx, mean_Y, pct_under, pct_over")

    rows = []
    i = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        Y = to_luminance_b709(frame)
        mean_y, pct_under, pct_over = frame_stats_from_luma(Y, args.black, args.white)
        print(f"{i}, {mean_y:.2f}, {pct_under:.2f}, {pct_over:.2f}")
        rows.append((i, mean_y, pct_under, pct_over))

        processed += 1
        i += 1

        # If user set a positive cap, stop there; if 0 or negative, process ALL frames
        if args.max_frames > 0 and processed >= args.max_frames:
            break
    
    print(f"\n# Processed {processed} frames total.")
    cap.release()

    if rows:
        # Choose ranking metric per frame
        ranked = []
        for (idx, mean_y, pu, po) in rows:
            if args.metric == "combined":
                score = pu + po
            elif args.metric == "over":
                score = po
            else:  # "under"
                score = pu
            ranked.append((idx, score))

        # Optional threshold filter
        ranked = [(idx, s) for (idx, s) in ranked if s >= args.min_clip]

        # Sort descending by score
        ranked.sort(key=lambda t: t[1], reverse=True)

        # How many to show
        k = len(ranked) if args.top <= 0 else min(args.top, len(ranked))
        label = {"combined": "total clipped (under+over)",
                "over": "overexposed",
                "under": "black-crushed"}[args.metric]

        print(f"\n# Worst {k} frames by {label} (>= {args.min_clip}%):")
        for idx, score in ranked[:k]:
            # Optional timestamp if you want it:
            # time_sec = idx / fps if fps else 0
            # print(f"  frame {idx:>6} @ {time_sec:7.3f}s : {score:6.2f}%")
            print(f"  frame {idx}: {score:.2f}%")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            wtr = csv.writer(f)
            wtr.writerow(["frame_idx", "mean_Y", "pct_under", "pct_over"])
            for (idx, m, pu, po) in rows:
                wtr.writerow([idx, f"{m:.4f}", f"{pu:.4f}", f"{po:.4f}"])
        print(f"\n# Wrote CSV: {args.csv}")

if __name__ == "__main__":
    main()
