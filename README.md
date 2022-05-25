# Rigid Multiblobs in half-space

Rotational and Translational Diffusion of Confined Rigid Bodies
by Steven Delong, Florencio Balboa, Blaise Delmotte, Brennan Sprinkle
and Aleksandar Donev (donev@courant.nyu.edu)
Courant Institute of Mathematical Sciences.

This package contains several python codes to run simulations of 
rigid bodies made out of  rigidly connected _blobs_, and confined above a
a single wall (floor). These codes can compute the
mobility of complex shape objects, solve mobility or resistance problems
for suspensions of many bodies or run deterministic or stochastic 
dynamic simulations. **If the object happens to be a sphere, we also provide 
a faster and more accurate alternative to the rigid multiblob method, which uses Lubrication corrections 
to improve the performance of nearfield hydrodynamic calculations.** 

For the theory behind the numerical methods consult the references:

1. **Brownian Dynamics of Confined Rigid Bodies**, S. Delong, F. Balboa Usabiaga, and A. Donev,
The Journal of Chemical Physics, **143**, 144107 (2015). 
[DOI](http://dx.doi.org/10.1063/1.4932062) [arXiv](http://arxiv.org/abs/1506.08868)

2. **Hydrodynamics of suspensions of passive and active rigid particles: a
rigid multiblob approach** F. Balboa Usabiaga, B. Kallemov, B. Delmotte,
A. Pal Singh Bhalla, B. E. Griffith, and A. Donev, 
Communications in Applied Mathematics and Computational Science,
**11**, 217 (2016). 
[DOI](http://dx.doi.org/10.2140/camcos.2016.11.217) [arXiv](http://arxiv.org/abs/1602.02170)

3. **Brownian dynamics of condined suspensions of active microrollers**, F. Balboa Usabiaga, B. Delmotte and A. Donev,
The Journal of Chemical Physics, **146**, 134104
(2017). [DOI](http://dx.doi.org/10.1063/1.4979494)
[arXiv](https://arxiv.org/abs/1612.00474)

4. **Large Scale Brownian Dynamics of Confined Suspensions of Rigid
Particles**, B. Sprinkle, F. Balboa Usabiaga, N. Patankar and
A. Donev, The Journal of Chemical Physics, **147**, 244103 (2017)
[DOI](http://dx.doi.org/10.1063/1.5003833)
[arXiv](https://arxiv.org/abs/1709.02410)

5. **Driven dynamics in dense suspensions of microrollers** by B. Sprinkle, E. B. van der Wee and Y. Luo and M. Driscoll, and A. Donev, in press in Soft Matter, 2020
[DOI](http://dx.doi.org/10.1039/D0SM00879F) [arXiv](https://arxiv.org/abs/2005.06002)

6. **Reconfigurable microbots folded from simple colloidal chains** by T. Yang, B. Sprinkle, Y. Guo, J. Qian, D. Hua, A. Donev, D. W.M. Marr, and N. Wu, PNAS, 202007255, 2020 [DOI](http://dx.doi.org/10.1073/pnas.2007255117)

Several example scripts for simulating immersed rigid bodies near a single
wall are present in subfolders.

For usage see **doc/README.md**.

### Software organization
* **doc/**: documentation.
* **body/**: it contains a class to handle a single rigid body.
* **Lubrication/** A small class which implements our fast lubrication corrected sheme for fluctuating colloidal suspensions above a wall. Instalation instructions are included in `./Lubrication/README.md`. The directory `./Lubrication/Lubrication_Examples` contains an example from [5], see documentation in `./Lubrication/Lubrication_Examples/Uniform_Rollers/README.md` as well as [6], see documentation `./Lubrication/Lubrication_Examples/Magnetic_Chain_With_Twist/README.md`. 
* **cRigid_cFibers/** Two small classes which implement our rigid multiblob method in a fast C++ code, and our flexible fibers method in a similar C++ code.
* **boomerang/**: older stochastic example from [1], see documentation `boomerang/README.md`.
* **sphere/**: the folder contains an example to simulate a sphere
whose center of mass is displaced from the geometric center
(i.e., gravity generates a torque), sedimented near a no-slip wall
in the presence of gravity, as described in Section IV.C in [1].
Unlike the boomerang example this code does not use a rigid
multiblob model of the sphere but rather uses the best known
(semi)analytical approximations to the sphere mobility.
See documentation `doc/boomerang.txt`.
* **many_bodyMCMC/**: Markov Chain Monte Carlo code for rigid bodies.
* **mobility/**: it has functions to compute the blob mobility matrix **M** and the
product **Mf** using CPUs or GPUs, see [2].
* **multi_bodies/**: codes to run many-body simulations, based on [3] (minimally-resolved active rollers) and primarily on [4] (general many-particle case).
* **quaternion_integrator/**: it has a small class to handle quaternions and
the schemes to integrate the equations of motion, see [1] and [4].
* **stochastic_forcing/**: it contains functions to compute the product
 **M**^{1/2}**z** necessary to perform Brownian simulations, see [3] and [4].
* **utils.py**: this file has some general functions that would be useful for
general rigid bodies (mostly for analyzing and reading trajectory
data and for logging).
