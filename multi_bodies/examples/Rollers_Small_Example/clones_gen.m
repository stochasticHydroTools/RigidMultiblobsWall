clc
close all


mg = 0.0024892;
kt = 0.0041419464;
a = 0.656;
N = 63;
y1 = (a:2.1*a:2.1*a*N)';
x1 = 0*y1;

x = [];
y = [];
num_lines=16;
for k = 1:num_lines
x = [x;x1+2.2*a*k];
y = [y;y1];
end
z = a+(kt/mg)+0*x;

disp(['use Lp = ' num2str(max(y)+a) ' as the periodic distance in y'])

plot(x,y,'o')

configs_file = ['./' num2str(num_lines) '_lines.clones'];
dlmwrite(configs_file,length(x),'delimiter','\t','precision',5)
dlmwrite(configs_file,[x y z 0*z+1 0*z 0*z 0*z],'-append','delimiter','\t','precision',12)