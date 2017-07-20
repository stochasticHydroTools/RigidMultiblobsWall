%Author: Zhe Chen
%2D Brownian motion with gaussian distrition at initial

tic;
%Parameters
n=4096;%number of particles
a=0.656;%radius of particles
sigma0=sqrt(n/2)*a;%initial sigma of gaussian distribution
H=5;%height of balaced plane
eta=1.0e-3;%viscocity
dt=0.08;%Time step $\Delta t$.
n_steps=200;%How many time steps to iterate.
sample=20;%Prob Distribution will be recorded at every 'sample' time steps.
repeat=2;%The procedure will be ran 'repeat' times.
r_step_hori=10;%r interval in the horizontal direction to sample on $ln(p)\sim r^2$
r_num_hori=20;%The number of r\_step in the horizontal direction  where we calculate the P(r).
r_step_vert=0.2;%r interval in the vertical direction to sample on $ln(p)\sim r^2$
r_num_vert=10;%The number of r\_step in the vertical direction  where we calculate the P(r).
% n=1024*4;%Particles' number, which is the same with 'brownian\_walker' on github. 
% D=1;%The diffusion coefficient.
output_name='data/blobswall';
echo=0;%echo=1: print all the iteration info; 0: not
kT=0.0165677856;%k_BT
k=0.0165677856;%stiffness
tau=6*pi*a*eta/k %relaxation time
hydro_interaction=1; %1: with HI; 0 without HI
plot_no_log=1;%whether plot Prob Distribution without log
mu=(1-9*a/(16*H)+(2*a^3)/(16*H^3)-a^5/(16*H^5))/(6*pi*eta*a);
D=kT*mu;


%initialize
prob_hori=zeros(repeat,r_num_hori,floor(n_steps/sample)+1);
prob_vert=zeros(repeat,2*r_num_vert,floor(n_steps/sample)+1);
Sigma_hori=zeros(repeat,2,2,floor(n_steps/sample)+1);
Sigma_vert=zeros(repeat,floor(n_steps/sample)+1);
for iter=1:repeat
    iter
    toc
    %initial gaussian distribution
    initial(n,a,sigma0,H);
    command='python ../../multi_bodies/multi_bodies.py --input-file inputfile_hydroGrid.dat';
    if echo
        [status,cmdout] = system(command,'-echo') ;
    else
        [status,cmdout] = system(command) ;
    end
    if status~=0
        error(num2str(status));
    end
    for t=1:floor(n_steps/sample)+1
        [prob_hori(iter,:,t),Sigma_hori(iter,:,:,t)]=prob_gen_hori([output_name,'.init.',num2str((t-1)*sample,'%08u'),'.clones'],r_step_hori,r_num_hori,H);
        [prob_vert(iter,:,t),Sigma_vert(iter,t)]=prob_gen_vertical([output_name,'.init.',num2str((t-1)*sample,'%08u'),'.clones'],r_step_vert,r_num_vert,H);
    end
end

%save
save('data.mat','prob_hori','prob_vert','Sigma_hori','Sigma_vert');

%plot error bar
% error_bar_plot_vertical(prob_vert,r_step_vert,sample,dt,Sigma_vert)
% error_bar_plot_horizontal(prob_hori,r_step_hori,sample,dt,hydro_interaction,Sigma_hori,sigma0,D,plot_no_log)

