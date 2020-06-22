# LubricationCorrections
Fast algorithms for applying lubrication corrections to fluctuating suspensions of spherical particles above a bottom wall

## Prerequisites
In order to run the code, the SuiteSparse library must be installed on the system
`http://faculty.cse.tamu.edu/davis/suitesparse.html`

In Ubuntu 18, this can be accomplished by running
`sudo apt install libsuitesparse-dev`

## C++ dependencies
Additionally, the C++ libraries `Boost` and `Eigen` must be installed
Ensure that the verion of Boost is `>= 1.64`

In Ubuntu 18, these libraries can be installed by running, e.g.
```
sudo apt install libeigen3-dev
sudo apt install libboost1.67-all-dev
```
## Python dependencies
In addition, the following python packages must be installed via pip

```
pip install --user scipy
pip install --user pyamg
pip install --user scikit-sparse
```

## Makefile

To run the code, one needs to `Make` the C++ helper code in the directory `/Lubrication`
```
https://github.com/stochasticHydroTools/LubricationCorrections/tree/master/Lubrication
```
To ensure the `Makefile` functions properly, 

Change lines 


https://github.com/stochasticHydroTools/LubricationCorrections/blob/eb60857f0899393958628d1ee223733d567409b9/Lubrication/Makefile#L9-L10

to reflect the location of Boost on your system, and change line


https://github.com/stochasticHydroTools/LubricationCorrections/blob/eb60857f0899393958628d1ee223733d567409b9/Lubrication/Makefile#L12


to relect the location of the Eigen `Include` directory on your system.

NOT TESTED:
If python3 is to be used, change to `-lboost_python3 -lboost_numpy3` in line
https://github.com/stochasticHydroTools/LubricationCorrections/blob/eb60857f0899393958628d1ee223733d567409b9/Lubrication/Makefile#L26

## Test
To test the the C++ helper function run properly, simply run 
```
python test_Lub_Class_CC.py 
```
after a sucessful `make` in the `/Lubrication` directory. If this code produces output, 
then things are working as they should. 


