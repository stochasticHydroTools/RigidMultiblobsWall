clear all
close all


Npx = 20;
Npy = 100;
Np = Npx*Npy;
a =0.1913417161825449;
dx = 2.2*a;
dx_rand = 0.1*a;
dy = 2.2*a;
dy_rand = dx_rand;
z = 1.2*a;

pos = zeros(Np,3);
pos(:,3) = z;
quat = zeros(Np,4);
quat(:,1) = 1;

for i = 1:Npx
    for j = 1:Npy
           pos((i-1)*Npy+j,1) = (i-1)*dx + rand*dx_rand;
           pos((i-1)*Npy+j,2) = (j-1)*dy + rand*dy_rand;
    end
end


pos(:,1) = pos(:,1) - mean(pos(:,1));
pos(:,2) = pos(:,2) - mean(pos(:,2));


theta = linspace(0,2*pi,100);


figure
hold on
for n=1:Np
    plot(pos(n,1)+a*cos(theta),pos(n,2)+a*sin(theta),'-k')
end
axis equal
grid on
box on

folder = 'Structures';
filename =  ['Lattice_blobs_Np_' num2str(Np) ...
             '_a_' strrep(num2str(a),'.','_') ...
             '_Z_' strrep(num2str(z/a),'.','_') 'a'...
             '_dx_dy_'  strrep(num2str(dx/a),'.','_') '_' strrep(num2str(dy/a),'.','_') 'a' ]
         
dlmwrite([folder '/' filename '.clones'],[pos quat], ' ')
