%Author: Zhe Chen
%2D Brownian motion with gaussian distrition at initial

tic;

%Parameters
dt=1;%Time step $\Delta t$
sigma0=90;%initial sigma of gaussian distribution.
t_steps=1000;%How many time steps to iterate.
sample=100;%Prob Distribution will be recorded at every 'sample' time steps.
repeat=500;%The procedure will be ran 'repeat' times.
r_step=10;%r interval to sample on $ln(p)\sim r^2$
r_num=10;%The number of r\_step where we calculate the P(r).
N=1024*16;%Particles' number, which is the same with 'brownian\_walker' on github. 
D=1;%The diffusion coefficient.

%initialize
prob=zeros(repeat,r_num,floor(t_steps/sample));

for iter=1:repeat
    if mod(iter,100)==0
        toc
        iter
    end
    %initial gaussian distribution
    loc=randn(N,2)*sigma0;
    for t=1:t_steps
        if mod(t-1,sample)==0
            %scatter(loc(:,1),loc(:,2),1,'black','filled');
            
            %calculate prob distribution
            prob(iter,:,(t-1)/sample+1)=prob_gen(loc,r_step,r_num);
%             std(loc)
        end
        
        %update
        loc=loc+sqrt(2*D*dt)*randn(N,2) ;
    end
end

%save
save('prob.mat','prob');

%plot error bar
error_bar_plot(prob,r_step,sample,dt,sigma0,D);

    
