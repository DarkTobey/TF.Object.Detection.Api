
ffmpeg -threads 2  -y -r 10 -i result/%%d.jpg -vcodec libx264 result.mp4