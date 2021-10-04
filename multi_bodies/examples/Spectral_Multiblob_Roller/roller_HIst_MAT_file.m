%clear all
%close all

set(0,'defaulttextInterpreter','latex')
set(0, 'defaultAxesTickLabelInterpreter','latex'); 
set(0, 'defaultLegendInterpreter','latex');
set(0, 'defaultLineLineWidth',3);
set(0,'defaultAxesFontSize',35)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% V_y = dlmread('./Exp_Data/displacements_y_2_first_bin2x_dt_10.dat');
% V_x = dlmread('./Exp_Data/displacements_x_2_first_bin2x_dt_10_in_pixels.dat');
% theta = asin(mean(V_y)/mean(sqrt(V_x.^2 + V_y.^2)));
% R = [cos(theta),sin(theta);-sin(theta),cos(theta)];
% V_xy = [V_x'; V_y'];
% V_new = R*V_xy;
% V_x = V_new(1,:);

V_y = dlmread('./Exp_Data/Vel_y_Second.dat');
V_x = dlmread('./Exp_Data/Vel_x_Second.dat');

NBINS = 40;
%NBINS = 0:1.4:85;
his = histogram(V_x,NBINS,'BinLimits',[0 82]);%hist(V_new(1,:),NBINS);
h = his.Values;
be = his.BinEdges;
b = 0.5*(be(2:end) + be(1:end-1));
h = h./trapz(b,h);
figure(1)
hplot(1) = plot(b,h,'k');
hold all


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% subplot(1,3,2)
% [hy,by] =hist(V_new(2,:),Exp_Bins);
% plot(by,hy./trapz(by,hy),'k')
% xlabel('$$V_y$$')
% hold all

phi = 0.4;
fps = 1; %1; %1/10;
a = 1;
%close all
cols = parula(8);
for k = [1]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



a = 1.0155;
A = dlmread('./run_two_rollers/data.Const_Torque_t_15.config');
dt = 0.01
%MAT_FILE_NAME = './MAT_FILES/LOWEST_FINAL_3_TORQUE.mat';
MAT_FILE_NAME = './MAT_FILES/TORQUE_12_MB_COMPARE.mat';


n_bods = A(1,1)
A(1:(n_bods+1):end,:) = [];
A(1:(100*n_bods),:) = [];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%fps = dt;

skip = floor(fps/dt);
xvels = [];
yvels = [];
xvels_low = [];
xvels_high = [];

av_zs = [];

Lp = sqrt(pi*(a^2)*n_bods/phi); %0
for i = 1:((length(A)/n_bods)-skip)
    i_next = i+skip;
    xp = A((i-1)*n_bods+1:i*n_bods,1);
    yp = A((i-1)*n_bods+1:i*n_bods,2);
    zp = A((i-1)*n_bods+1:i*n_bods,3);
    x = A((i_next-1)*n_bods+1:i_next*n_bods,1);
    y = A((i_next-1)*n_bods+1:i_next*n_bods,2);
    z = A((i_next-1)*n_bods+1:i_next*n_bods,3);
    dist_x = (x-xp);
    dist_y = (y-yp);
    
    
    xvel = (1/dt/skip)*dist_x; %sqrt(dist_x.^2 + dist_y.^2);
    yvel = (1/dt/skip)*dist_y;
    
    z_av = reshape(A((i-1)*n_bods+1:i_next*n_bods,3),n_bods,skip+1);
    z_av = mean(z_av,2);
    
    av_zs = [av_zs; z_av(:)];
    
    xvels = [xvels; xvel(:)];
    yvels = [yvels; yvel(:)];
    xvels_low = [xvels_low; xvel(z_av < 2*a)];
    xvels_high = [xvels_high; xvel(z_av > 2*a)];
end

% V_H = [xvels av_zs];
% save('./MAT_FILES/FINAL_3_V_H.mat','V_H');



figure(1)

%b = linspace(-15,100,2*115);
[h,b] = hist(xvels,b);
normalize = trapz(b,h);
h = h./normalize;


%%%%%%%%%%%%%%%%%%%%%%%%
SAVE_V_b = b;
SAVE_V_h = h;
%%%%%%%%%%%%%%%%%%%%%%%%


%plot([b(b>0) -Lp*0.5+(b(b>0))],[h(b>0) (h(b<0))],'color',cols(k,:))
hplot(2) = plot(b,h,'color',cols(k,:));
hold all
mn = mean(xvels);
%plot([mn mn],[0 1.2*max(h)],':','color',cols(k,:),'linewidth',3)
%hold all
[h,b] = hist(xvels_low,b);
[m_low,I_low] = max(h);
low_peak = b(I_low);
hplot(3) = plot(b,h./normalize,':','color',cols(k+2,:));
hold all
plot([low_peak low_peak],get(gca,'ylim'),':','color',cols(k+2,:),'linewidth',1.5)
hold all

%%%%%%%%%%%%%%%%%%%%%%%%
SAVE_V_b_low = b;
SAVE_V_h_low = h./normalize;
%%%%%%%%%%%%%%%%%%%%%%%%


[h,b] = hist(xvels_high,b);
[m_high,I_high] = max(h);
high_peak = b(I_high);
hplot(4) = plot(b,h./normalize,'--','color',cols(k+4,:));
hold all
plot([high_peak high_peak],get(gca,'ylim'),'--','color',cols(k+4,:),'linewidth',1.5)
xlabel('$$V_x$$')
ylabel('$$P(V_x)$$')
title(['fps = ' num2str(1/fps)])
xlim([0 70])

%%%%%%%%%%%%%%%%%%%%%%%%
SAVE_V_b_high = b;
SAVE_V_h_high = h./normalize;
%%%%%%%%%%%%%%%%%%%%%%%%


%ylim([0 1.2*max(h)])
%xticks(sort([-5 0 10 20 floor(mn*100)/100]))
%xt = get(gca,'xtick');
xt = [0 30 70];
xt = sort([xt floor(high_peak*100)/100 floor(low_peak*100)/100]);
xticks(xt)

bz = linspace(0,6,150);
[hz,bz] = hist(A(:,3),bz);
hz = hz./trapz(bz,hz);

%%%%%%%%%%%%%%%%%%%%%%%%
SAVE_Z_b = bz;
SAVE_Z_h = hz;
%%%%%%%%%%%%%%%%%%%%%%%%


% save(MAT_FILE_NAME,'SAVE_Z_b','SAVE_Z_h','SAVE_V_h_high','SAVE_V_b_high',...
%                    'SAVE_V_h_low','SAVE_V_b_low','SAVE_V_h','SAVE_V_b')


end

%%
% cfgs = A((i_next-1)*n_bods+1:i_next*n_bods,:);
% phi = 0.4;
% L = sqrt(pi*(a^2)*n_bods/phi)
% cfgs_p = cfgs;
% for i = 1:n_bods
%     for k = 1:2
%         while cfgs_p(i,k) > L
%             cfgs_p(i,k) = cfgs_p(i,k)-L;
%         end
%         while cfgs_p(i,k) < 0 
%             cfgs_p(i,k) = cfgs_p(i,k)+L;
%         end
%     end
% end
% plot3(cfgs_p(:,1),cfgs_p(:,2),cfgs_p(:,3),'.','markersize',18)
% daspect([1 1 1])
% 
% f_name = '/home/hat/Misc_Codes/Test_Data_For_Rollers/Const_Torque_t_15.clones';
% dlmwrite(f_name,n_bods,'delimiter','\t','precision',5)
% dlmwrite(f_name,cfgs_p,'-append','delimiter','\t','precision',12)


