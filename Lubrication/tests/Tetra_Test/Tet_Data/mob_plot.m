clc
%close all

set(0,'defaulttextfontsize',25);
set(0,'defaultaxesfontsize',25);
set(0,'defaultaxeslinewidth',3);
set(0, 'DefaultLineLineWidth',3);
set(0, 'defaultAxesTickLabelInterpreter','latex'); 
set(0, 'defaultLegendInterpreter','latex');
set(0,'defaultTextInterpreter','latex');

MB_Lub_pyb = dlmread('tetra_lub_pybind.dat');
MB642 = dlmread('tetra_mb642_mob.dat');
MB2562 = dlmread('tetra_mb2562_mob.dat');
epsilon = MB2562(:,1);

normsLub = 0*epsilon;
norms642 = 0*epsilon;
for i = 1:length(epsilon)
    Mob_Lub = reshape(MB_Lub_pyb(i,2:end),4*6,4*6);
    Mob_642 = reshape(MB642(i,2:end),4*6,4*6);
    Mob_2562 = reshape(MB2562(i,2:end),4*6,4*6);
    
    normsLub(i) = norm(Mob_2562-Mob_Lub)/norm(Mob_2562);
    norms642(i) = norm(Mob_2562-Mob_642)/norm(Mob_2562);
end

base = hex2rgb('#833ab4');
middle = hex2rgb('#fd1d1d');
last = hex2rgb('#fcb045');

t = (linspace(0,1,3));
PC_cols = t'*middle + (1-t')*base;%round(0.8*base);
PC_cols = [PC_cols; t'*last + (1-t')*middle];
PC_cols = flipud(PC_cols);

fig = loglog(epsilon-1, normsLub, '-','color',PC_cols(1,:),'linewidth',5);
hold all
loglog(epsilon-1,norms642,'k','linewidth',6)
leg = legend('Lubrication','642 blobs');
set(leg,'fontsize',25)
xlim([0.01 2.5])
% xt = get(gca,'xtick');
% xt = set(gca,'xtick',sort([xt, 0.01]));
xlabel('$$\epsilon$$')
ylabel('Relative Error in Mobility')

set(gcf, 'position', [100, 100, 1100, 900])

saveas(fig, "lub_tetra_test.png");

%print('-depsc','-r300',['/home/bs162/GIT_Directories/GIT_papers/Papers/Lubrication/figures/Tetra_Mobility_Error.eps'])
