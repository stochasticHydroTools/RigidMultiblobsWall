### Probability Distribution parallel or verticle to the wall with or without Hydrodynamics Interaction(HI)

#### 1. Run the 'err_bar_main.m' and it will initialize particles in gaussian distribution and doing simulations for given times.

#### 2. Figures would be automatically saved in 'fig/' and prob distribution would be saved in 'data.mat'. Variance is saved in 'sigma_vertical' and 'sigma_hori'. 

**Notice:** Since matlab would crush somehow with too much ploting operations, I commended the 'plot' line in the last 2 lines of 'err_bar_main.m'. You can uncommend it or just execute the two lines after the program runs successfully.

#### 3. Please see 'report/ProbDist.pdf' to see the report that contains theoretic analysis, simulation results and discussions in detail.

#### 4. Some primary paramenters with defaults (listed here for reference, no need to read it.) :

##### 1. In err_bar_main.m

n=4096;%number of particles

a=0.656;%radius of particles

H=5;%height of balaced plane

eta=1.0e-3;%viscocity

dt=0.02;%Time step $\Delta t$.

n_steps=800;%How many time steps to iterate.

sample=80;%Prob Distribution will be recorded at every 'sample' time steps.

repeat=50;%The procedure will be ran 'repeat' times.

r_step_hori=3;%r interval in the horizontal direction to sample on $ln(p)\sim r^2$

r_num_hori=20;%The number of r\_step in the horizontal direction  where we calculate the P(r).

r_step_vert=0.1;%r interval in the vertical direction to sample on $ln(p)\sim r^2$

r_num_vert=10;%The number of r\_step in the vertical direction  where we calculate the P(r).

echo=0;%echo=1: print all the iteration info; 0: not

kT=0.0165677856;%k_BT

k=0.0165677856*4;%stiffness

tau=6*pi*a*eta/k %relaxation time

hydro_interaction=0; %1: with HI; 0 without HI

plot_no_log=1;%whether plot Prob Distribution without log

##### 2. In inputfile_hydroGrid.dat:

hydro_interactions   0

repulsion_strength_wall                  0.0662711424

debye_length_wall	   		                 5.0

dt		 	   	     	                       0.02

n_steps					                         800

n_save  				                         80

eta                                      1.0e-3

g					                               0.0

blob_radius				                       0.656

kT					                             0.01656778564
                                
periodic_length				                   0 0 0