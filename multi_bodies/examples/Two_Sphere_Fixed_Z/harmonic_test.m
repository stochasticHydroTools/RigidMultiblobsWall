clc
close all

cd '/home/bs162/Sedimentation/RigidMultiblobsWall_NoZcompTest/multi_bodies/examples/Two_Sphere_Fixed_Z'

N = 2;
A = dlmread('two_sphere_test.N_2_tilt_1.047.config');
A(1:N+1:end,:) = [];
R1 = A(1:N:end,1:3);
R2 = A(2:N:end,1:3);
R = R2-R1;

r = sqrt(sum(R.^2,2));

[h,b] = hist(r,100);
H = h./trapz(b,h);

plot(b,H)
hold all
f = @(x) exp(-(b-8.4).^2);
plot(b,f(b)./trapz(b,f(b)))