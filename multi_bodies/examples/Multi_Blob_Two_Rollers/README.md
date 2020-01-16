This example shows two 'multi blob' rollers which are translating in a hydrodynamically bound state. The hydrodynamic radius of the rollers is set to `R_h = 1` for simplicity and the resolution can be changed by changing line 64 in `inputfile_rollers.dat` from 
`structure	../../Structures/shell_N_12_Rg_0_7921_Rh_1.vertex two_rollers.clones`
to, say
`structure	../../Structures/shell_N_42_Rg_0_8913_Rh_1.vertex two_rollers.clones`

AND 

changing line 41 to the corresponding value for `blob_radius`. e.g changing

`blob_radius				                     0.41642068`
to
`blob_radius				                     0.2435531`
in the previous example. 

A stochastic or deterministic simulation can be done by swapping the comment from lines 9 and 10. 

To run a deterministic simulation using 12 blobs per roller, simply execute 

`python multi_bodies.py --input-file inputfile_rollers.dat`

The result of a simulation is saved in the directoy `./run_two_rollers` and can be visualized using the MATLAB file `plot_config.m`

To save `.png` images of the movie,  set `print_pngs = 1;` at line 15 of `plot_config.m` -- they will be saved in a directory called `roller_pngs` in the `Multi_Blob_Two_Rollers` directory.

On Linux make a movie from the PNGs by going into the directory `roller_pngs` and executing:

`ffmpeg -framerate 3 -i rollers_%d.png -pix_fmt yuv420p -r 3 -vcodec libx264 -an -vf scale=1800:900 movie.mp4`
