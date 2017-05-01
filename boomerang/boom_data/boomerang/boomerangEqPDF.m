len = 300001;
burn = 50000;

path = '/home/bsprinkle/Research/RotationalDiffusion-git/RotationalDiffusion-private/boomerang/boom_data/boomerang/';
for i = 1:4
boom = [path 'boomerang-trajectory-dt-0.01-N-300001-scheme-FIXMAN-g-1.0-ex-' num2str(i) '.txt'];
fileID = fopen(boom,'r');
A = textscan(fileID,'%f %f %f %f %f %f %f','Delimiter',',', 'HeaderLines', 15, 'CollectOutput', 1);
fclose(fileID);
B = A{1};
b(1+(i-1)*(len-burn):i*(len-burn)) = B(burn+1:end,3);
end
figure(1)
[f1,x] = hist(b,1000);
F1 = f1/trapz(x,f1);
bar(x,F1);

clear A B b
len = 300001;
path = '/home/bsprinkle/Research/RotationalDiffusion-git/RotationalDiffusion-private/Data_multi_body/data/';
for i = 1:4
boom = [path 'rod-trajectory-dt-0.01-N-300001-scheme-PC2-g-1.0-ETA1e-4-Yukawa-ex-' num2str(i) '.txt'];
fileID = fopen(boom,'r');
A = textscan(fileID,'%f %f %f %f %f %f %f','Delimiter',',', 'HeaderLines', 15, 'CollectOutput', 1);
fclose(fileID);
B = A{1};
b(1+(i-1)*(len-burn):i*(len-burn)) = B(burn+1:end,3);
end
figure(2)
[f2,x] = hist(b,1000);
F2 = f2/trapz(x,f2);
bar(x,F2);

percentNormalizedMSE = 100*sqrt(mean((F1-F2).^2))/(max(x)-min(x))