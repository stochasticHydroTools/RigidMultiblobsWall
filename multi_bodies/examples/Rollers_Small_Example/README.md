# Rollers example

Input files to simulate active rollers moving above a wall in the
presence of thermal fluctuations. For details see reference

1. **Brownian dynamics of condined suspensions of active microrollers**, F. Balboa Usabiaga, B. Delmotte and A. Donev,
The Journal of Chemical Physics, **146**, 134104 (2017). [arXiv](https://arxiv.org/abs/1612.00474)
[DOI](http://dx.doi.org/10.1063/1.4979494)

To use this example is necessary to have pyCuda installed and a GPU
compatible with CUDA. To run this example: 

Run the simulation with the command

```
python multi_bodies.py --input-file inputfile_rollers.dat
```

See options for the simulations in the input file
`inputfile_rollers.dat`.

To visualize, run the program `./run_rollers/plot_config.m` in MATLAB after data has been generated
in the directory `./run_rollers`.

Run the example to at least 700-800 steps to see the formation of a critter

To save `.png` images of the movie,  set `print_pngs = 1;` in `plot_config.m` -- they will be saved in a directory called `roller_pngs`.

On Linux make a movie from the PNGs by going into the directory `roller_pngs` and executing:

`ffmpeg -framerate 3 -i rollers_%d.png -pix_fmt yuv420p -r 3 -vcodec libx264 -an -vf scale=1800:900 movie.mp4`
