%Author: Zhe Chen
%Creat initial.dat at ./data/
function initial(n,a,sigma0,H)
%Creat initial.dat at ./data/
%-----------------------------
%Input:
%n: number of particles
%a: radius of particles
%sigma0: initial sigma of gaussion distribution in the plane
%H:height of the plane
temp=zeros(n,7);
temp(:,1:2)=randn(n,2)*sigma0;
temp(:,3)=H*ones(n,1);
temp(:,4)=ones(n,1);


fileID = fopen('./data/initial.dat','w');
fprintf(fileID,[num2str(n),'\n']);
for i=1:n
    fprintf(fileID,'%12.8f   %12.8f   %12.8f   %12.8f   %12.8f   %12.8f   %12.8f\n',temp(i,:));
end
fclose(fileID);