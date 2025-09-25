import sys, os, cv2

def main():
    if len(sys.argv) < 2:
        print("Usage: python hello_frame.py <path_to_video>")
        sys.exit(2)
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(2)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: cannot open video. Try another MP4 (H.264).")
        sys.exit(1)

    fps   = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"Video: {path}")
    print(f"Size:  {w}x{h}")
    print(f"FPS:   {fps:.3f}")
    print(f"Frames:{count}")

    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Error: failed to read first frame.")
        sys.exit(1)

    cv2.imshow("First frame", frame)
    print("Press any key in the image window to close…")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
