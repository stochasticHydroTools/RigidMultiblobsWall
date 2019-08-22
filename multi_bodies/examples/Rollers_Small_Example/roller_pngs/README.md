To save `.png` images of the movie in this directory,  set `print_pngs = 1;` in `plot_config.m`.

On Linux make a movie from the PNGs by executing:

`ffmpeg -framerate 3 -i rollers_%d.png -pix_fmt yuv420p -r 3 -vcodec libx264 -an -vf scale=1800:900 movie.mp4`
