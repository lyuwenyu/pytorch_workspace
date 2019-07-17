import glob
import os

dirs = glob.glob('record_*')

for d in dirs:
    os.system(f'rm -rf {d}/images')
    os.system(f'mkdir {d}/images')
    os.system(f'ffmpeg -i {d}/*/video_camera_lka.h264 -q:v 2 -r 5 {d}/images/%5d.jpg')
    