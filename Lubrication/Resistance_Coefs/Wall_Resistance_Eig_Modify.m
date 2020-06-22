R2 = dlmread('mob_scalars_wall_MB_2562.txt');
Rmbw = dlmread('res_scalars_wall_MB.txt');
Rmbw = flipud(Rmbw);
h = R2(:,1);

Ri2 = 0*R2;
Ri2_clip = 0*R2;
M2_clip = 0*R2;

norm_cut = 0*h;

 first_h = 1;

for j = 1:length(h)
 
 %%%%%%%%%%%%%
 
Xa = R2(j,2);
Ya = R2(j,3);
Yb = R2(j,4);
Xc = R2(j,5);
Yc = R2(j,6);

denom = Ya*Yc - Yb*Yb;
RXa = 1.0/Xa;
RYa = Yc/denom;
RYb = -Yb/denom;
RXc = 1.0/Xc;
RYc = Ya/denom; 

Ri2(j,:) = [h(j) RXa RYa RYb RXc RYc];

Rsup2 = [(RYa), 0, 0, 0, RYb, 0;
	 0, (RYa), 0, -RYb, 0, 0;
	 0, 0, (RXa), 0, 0, 0;
	 0, -RYb, 0, RYc, 0, 0;
	 RYb, 0, 0, 0, RYc, 0;
	 0, 0, 0, 0, 0, RXc]; 
 %%%%%%%%%%%%%
 
mbXa = interp1(Rmbw(:,1), Rmbw(:,2), h(j), 'pchip'); 
mbYa = interp1(Rmbw(:,1), Rmbw(:,3), h(j), 'pchip'); 
mbYb = interp1(Rmbw(:,1), Rmbw(:,4), h(j), 'pchip'); 
mbXc = interp1(Rmbw(:,1), Rmbw(:,5), h(j), 'pchip'); 
mbYc = interp1(Rmbw(:,1), Rmbw(:,6), h(j), 'pchip'); 

Rmb = [(mbYa), 0, 0, 0, mbYb, 0;
	 0, (mbYa), 0, -mbYb, 0, 0;
	 0, 0, (mbXa), 0, 0, 0;
	 0, -mbYb, 0, mbYc, 0, 0;
	 mbYb, 0, 0, 0, mbYc, 0;
	 0, 0, 0, 0, 0, mbXc];
 

 Re2 = Rsup2-Rmb;
 [V,S] = eig(Re2);
 S(S<=1e-10) = 1e-10;
 Delta_clip = V*S*V';
 Re2_clip = Delta_clip+Rmb;
 
 
 Xa = Re2_clip(3,3);
 Ya = Re2_clip(1,1);
 Yb = Re2_clip(1,5);
 Xc = Re2_clip(6,6);
 Yc = Re2_clip(5,5);
 
 denom = Ya*Yc - Yb*Yb;
 RXa = 1.0/Xa;
 RYa = Yc/denom;
 RYb = -Yb/denom;
 RXc = 1.0/Xc;
 RYc = Ya/denom; 
 
 Ri2_clip(j,:) = [h(j) Xa Ya Yb Xc Yc];
 M2_clip(j,:) = [h(j) RXa RYa RYb RXc RYc];
 
 norm_cut(j) = norm(Re2_clip-Rsup2);
 
 
end


epsilon = h-1;

% Xa_asym = 1.0./epsilon - (1.0/5.0)*log(epsilon) + 0.971280;
% Ya_asym = -(8.0/15.0)*log(epsilon) + 0.9588;
% Yb_asym = -(-(1.0/10.0)*log(epsilon)-0.1895) - 0.4576*epsilon;
% Yb_asym = Yb_asym*4./3.;
% Xc_asym = 1.2020569 - 3.0*(pi*pi/6.0-1.0)*epsilon;
% Xc_asym = Xc_asym*4./3.;
% Yc_asym = -2.0/5.0*log(epsilon) + 0.3817 + 1.4578*epsilon;
% Yc_asym = Yc_asym*4./3.;


Xa_asym = 1.0./epsilon - (1.0/5.0)*log(epsilon) + 0.971280;
Ya_asym = -(8.0/15.0)*log(epsilon) + 0.9588;
Yb_asym = -(-(1.0/10.0)*log(epsilon)-0.1895) - 0.4576*epsilon;
Yb_asym = Yb_asym*4./3.;
Xc_asym = 1.2020569 - 3.0*(pi*pi/6.0-1.0)*epsilon;
Xc_asym = Xc_asym*4./3.;
Yc_asym = -2.0/5.0*log(epsilon) + 0.3817 + 1.4578*epsilon;
Yc_asym = Yc_asym*4./3.;

Yb_asym_old = -(-(1.0/10.0)*log(epsilon)-0.1895);
Yb_asym_old = Yb_asym_old*4./3.;

Yc_asym_old = -2.0/5.0*log(epsilon) + 0.3817;
Yc_asym_old = Yc_asym_old*4./3.;

Oneil_hoa = [1.003202 1.005004 1.0453 1.1276];
% table 1
Oneil_TF = -[-4.0223 -3.7863 -2.6475 -2.1514];
Oneil_TT = -(4./3.)*[3.8494e-1 3.4187e-1 1.4552e-1 7.3718e-2]; 
%multiply by (4/3) to match our version which is normalized by 6*piinstead
%of 8*pi as was done in the paper

% table 2
Oneil_RF = -[5.1326e-1 4.5582e-1 1.9403e-1 9.8291e-2];
Oneil_RT = -(4./3.)*[-2.6793 -2.5056 -1.6996 -1.3877];

asym_co = [h Xa_asym Ya_asym Yb_asym Xc_asym Yc_asym];


 figure(1)
 
 clf
 cols = lines(7);
 for i = 2:6
     subplot(2,3,i-1)
     plot(Ri2_clip(:,1)-1,Ri2_clip(:,i),'color',[63 94 251]./255,'linewidth',3) %cols(i,:)
     hold all
%      plot(Ri2(:,1),Ri2(:,i),':','color',cols(i,:))
%      hold all
     plot(asym_co(:,1)-1,asym_co(:,i),'--','color',[252,70,107]./255,'linewidth',3)
     if i == 4
        plot(asym_co(:,1)-1,Yb_asym_old,':','color',[252,0,0]./255,'linewidth',3) 
     end
     if i == 6
        plot(asym_co(:,1)-1,Yc_asym_old,':','color',[252,0,0]./255,'linewidth',3)  
     end
     set(gca,'linewidth',3)
     mks = 10;
     switch i
         case 2
             title('$$X^{tt}_{wall}$$')
             xlim([1.025 1.15]-1)
             ylim([5 50])
             xticks([1.01 1.05 1.1 1.15]-1)
             %yticks([0 25 50])
             xlabel('$$\epsilon_h$$')
             hold all
             cutt = 0.1;
             plot([cutt cutt],get(gca,'ylim'),'k-','linewidth',1)
         case 3
             title('$$Y^{tt}_{wall}$$')
             plot(Oneil_hoa-1,Oneil_TF,'o','markersize',mks,'color',0.0*[0.86 0.8 0.2],'markerfacecolor',[0.86 0.8 0.2])
             xlim([1.0001 1.15]-1)
             xticks([1.01 1.1 1.15]-1)
             yticks([2 4 6])
             xlabel('$$\epsilon_h$$')
             hold all
             cutt = 0.01;
             plot([cutt cutt],get(gca,'ylim'),'k-','linewidth',1)
         case 4
             title('$$Y^{tr}_{wall}$$')
             plot(Oneil_hoa-1,Oneil_TT,'o','markersize',mks,'color',0.0*[0.86 0.8 0.2],'markerfacecolor',[0.86 0.8 0.2])
             xlim([1.0001 1.3]-1)
             ylim([-1 0.1])
             xticks([1.01 1.1 1.2 1.3]-1)
             xlabel('$$\epsilon_h$$')
             hold all
             cutt = 0.1;
             plot([cutt cutt],get(gca,'ylim'),'k-','linewidth',1)
         case 5
             title('$$X^{rr}_{wall}$$')
             xlim([1.0001 1.02]-1)
             xticks([1.001 1.01 1.02]-1)
             yticks([1.55 1.575 1.6])
             xlabel('$$\epsilon_h$$')
             hold all
             cutt = 0.01;
             plot([cutt cutt],get(gca,'ylim'),'k-','linewidth',1)
         case 6
             title('$$Y^{rr}_{wall}$$')
             plot(Oneil_hoa-1,Oneil_RT,'o','markersize',mks,'color',0.0*[0.86 0.8 0.2],'markerfacecolor',[0.86 0.8 0.2])
             xlim([1.0001 1.2]-1)
             xticks([1.01 1.1 1.2 1.3]-1)
             %yticks([2 4 6])
             ylim([1 4.5])
             xlabel('$$\epsilon_h$$')
             hold all
             cutt = 0.1;
             plot([cutt cutt],get(gca,'ylim'),'k-','linewidth',1)
             leg = legend('2562 Multiblob','Assymptotic + Linear','Assymptotic','O''Neill Data','Cutoff')
             set(leg,'fontsize',35)
     end
 end
 
 print('-depsc','/home/bs162/GIT_Directories/GIT_papers/Papers/Lubrication/figures/wall_mobility_coefs.eps','-r300')
 
%  figure(2)
%  plot(h,norm_cut)
 
%  figure(2)
%  clf
%  for i = 2:6
%      semilogy(h,abs(Ri2_clip(:,i)-Ri2(:,i))./abs(Ri2(:,i)),'color',cols(i-1,:))
%      hold all
%  end
%  
%  figure(3)
%  clf
%  for i = 2:6
%      semilogy(h,abs(M2_clip(:,i)-R2(:,i))./abs(R2(:,i)),'color',cols(i-1,:))
%      hold all
%  end

% dlmwrite('mob_scalars_wall_MB_2562_eig_thresh.txt',M2_clip,'delimiter','\t','precision',12)
 