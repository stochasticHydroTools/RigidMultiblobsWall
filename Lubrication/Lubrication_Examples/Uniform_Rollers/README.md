# Microrollers
This example contains the code to reproduce each of the simulation curves from from figures 2 and 3b in
`Driven dynamics in dense suspensions of microrollers`
https://cims.nyu.edu/~donev/FluctHydro/RollersLubrication.pdf  

### Run the code
To run this example, ensure that the make file has been succesfully run in the `Lubrication` base directory.
Then one may simply type the following into a terminal 
```
python main_rollers.py --input-file inputfile_rollers_exp.dat
```
and data will be produced in the `./data` directory.

### Modifying the code
This example is set up to drive the rollers at a constant frequency but limit the amount of torque that can be applied to each particle. 
To change the driving mechanism to either an unmitigated, constant frequency or a constant torque, uncomment the requisite lines in
`main_rollers.py`
by following the comented instructions on lines:
-169,187 (For constant applied torque)
-181 (unmitigated, constant frequency) 

