# Magnetic chains
This example contains the code to reproduce the simulation data shown in figure 1 of the SI of
`Reconfigurable microbots folded from simple colloidal chains`
https://cims.nyu.edu/~donev/FluctHydro/ReconfigurableChains+SI.pdf

### Run the code
To run this example, ensure that the makefile has been succesfully run in the `Lubrication` base directory.
Then one may simply type the following into a terminal
```
python main_chain.py --input-file inputfile_chain.dat
```
and data will be placed in the `./data` directory

### Modify the code
Various physical properties of the colloidal chain can be modified by changing the following lines of `main_chain.py` 
-163 (strength of the magnetic force)
-177 (bending modulus)
-178 (twisting modulus)

and various properties of the driving B-field can be modified by changing the following lines of `main_chain.py`
-181 (angle of the rotating B-field)
-182 (frequency of the B-field)

To change the initial shape of the chain, modify the second argument (currently `./chain_60.clones`) in `line 39` of `inputfile_chain.dat`.
We've provided three initial shapes: 
-chain_60.clones (a straight line)
-hairpin_60.clones (a hairpin shape)
-S_curve_60.clones (an `S` shape)

The MATLAB code to produce these shapes is included in the file `chain_gen.m`

### Plot the results
To plot the results, simply change `f_name` on line 10 of the MATLAB file `chain_plot.m` to the desired config file, and run in MATLAB.
Uncomment line `91` in `chain_plot.m` to save sequential .png images which can be converted to a movie 
