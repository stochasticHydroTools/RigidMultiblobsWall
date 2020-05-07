clear all
close all


Nblobs = 1536;
Ndofs = 6;



folder = 'Structures';
filename =  ['slip_zero_N' num2str(Nblobs)]
         
dlmwrite([folder '/' filename '.dat'],zeros(Nblobs,Ndofs), ' ')
