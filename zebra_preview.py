import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Live zebra preview (OpenCV window).")
    p.add_argument("--video", required=True)
    p.add_argument("--mode", choices=["Over","Under","Both"], default="Both")
    p.add_argument("--black", type=int, default=16)
    p.add_argument("--white", type=int, default=235)
    p.add_argument("--scale", type=float, default=0.5, help="Preview scale (0.25..1.0)")
    p.add_argument("--phase-step", type=int, default=2)
    return p.parse_args()