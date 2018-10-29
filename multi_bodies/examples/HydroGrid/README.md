# HydroGrid 
This example shows how to set the inputfiles to call HydroGrid and
compute several structures factors. 

## Prepare the code
First, you need to download and compile HydroGrid, see

```
https://github.com/stochasticHydroTools/HydroGrid
```

Then copy the code `RigidMultiblobsWall/multi_bodies/multi_bodies.py`
to this folder and edit the lines 

```
# Add path to HydroGrid and import module
# sys.path.append('../../HydroGrid/src/')
```

to add the path to HydroGrid library. Instead of editing
`multi_bodies.py` you can copy the file `libCallHydroGrid.so` from
HydroGrid to this folder. 

## Input files
This example needs two inputfiles, `inputfile.dat` with the options
for `multi_bodies.py` and `hydroGridOptions.nml` with the options for
HydroGrid. The new options in `inputfile.dat` related with HydroGrid
are

* `call_HydroGrid`: (string (default `False`)) if True and the library
`HydroGrid` is available it will call `HydroGrid` to compute several structure
factors. See section 1.4 to see how to install HydroGrid. This option
should be used with periodic domains in the xy plane (see option
`periodic_length`). You need to have a file `hydroGridOptions.nml` in
the folder where you run the simulation. See the example in
`multi_bodies/examples/HydroGrid/`. 

* `sample_HydroGrid`: (int (default 1)) call HydroGrid every
`sample_HydroGrid` steps to sample the system.

* `save_HydroGrid`: (int (default 0)) save HydroGrid information every
`save_HydroGrid` steps. The average restarts after saving the data.
If 0 the conde only saves the final information. Use 0 if you want to
average over the whole simulations.

* `cells`: (two ints (default 1 1)) The blobs location define a 
discrete concentration field in the xy plane c(x,y) with `cells`
number of cells along the x and y axes. This concentration is passed
to HydroGrid to compute several structures factors.

* `green_particles`: (two ints (default 0 0)) Blobs with index
`green_particles[0] <= index < green_particles[1]` are labeled green,
the others are labelled red. This option has not any effect on the
dynamics of the simulation but it is used by HydroGrid to compute
several structures factors. 


The options for `hydroGridOptions.nml` and the new output files are
explained in `https://github.com/stochasticHydroTools/HydroGrid`. 

## Run the simulation
You just need to run the command

```
python multi_bodies.py --input-file inputfile.dat
```


