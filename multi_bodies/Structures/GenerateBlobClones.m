n = 5;
N = n^2;
phi = 0.25;
a = 0.656;

L = sqrt(pi*a^2*N/phi)

x = linspace(-L/2,L/2,n+1);
x = x(1:end-1);

[X,Y] = meshgrid(x,x);

locs = [X(:) Y(:) ones(N,1)*[1.5 1 0 0 0]];

path = '/home/bsprinkle/Research/RotationalDiffusion-git/BlobWall-Test/multi_bodies/Structures/';
fid = fopen([path 'blob_' num2str(N) '_phi_' num2str(phi) '.clones'],'w');
fprintf(fid,'%s\n',num2str(N));
for k = 1:N
    fprintf(fid,'%12.14f\t%12.14f\t%12.14f\t%12.14f\t%12.14f\t%12.14f\t%12.14f\n',locs(k,:));
end
fclose(fid);