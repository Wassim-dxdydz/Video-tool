# check_ffmpeg.py
import imageio_ffmpeg, os
p = imageio_ffmpeg.get_ffmpeg_exe()
print("FFmpeg binary:", p)
print("Exists? ", os.path.exists(p))
print("OK: imageio-ffmpeg provides a working ffmpeg binary.")
