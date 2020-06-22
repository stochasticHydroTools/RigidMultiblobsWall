R10 = dlmread('mob_scalars_wall_MB_10242_running.txt');
R2 = dlmread('mob_scalars_wall_MB_2562.txt');
Rmbw = dlmread('res_scalars_wall_MB.txt');
Rmbw = flipud(Rmbw);
h = R10(:,1);

Ri10 = 0*R2;
Ri2 = 0*R2;

meig10 = 0*h;
meig2 = 0*h;

 first_h = 1;

for j = 1:length(h)

    
%%%%%%%%%%%%%%
Xa = R10(j,2);
Ya = R10(j,3);
Yb = R10(j,4);
Xc = R10(j,5);
Yc = R10(j,6);

denom = Ya*Yc - Yb*Yb;
RXa = 1.0/Xa;
RYa = Yc/denom;
RYb = -Yb/denom;
RXc = 1.0/Xc;
RYc = Ya/denom; 

Ri10(j,:) = [h(j) RXa RYa RYb RXc RYc];

Rsup10 = [(RYa-1.), 0, 0, 0, RYb, 0;
	 0, (RYa-1.), 0, -RYb, 0, 0;
	 0, 0, (RXa-1.), 0, 0, 0;
	 0, -RYb, 0, RYc-4.0/3.0, 0, 0;
	 RYb, 0, 0, 0, RYc-4.0/3.0, 0;
	 0, 0, 0, 0, 0, RXc-4.0/3.0];
 
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

Rsup2 = [(RYa-1.), 0, 0, 0, RYb, 0;
	 0, (RYa-1.), 0, -RYb, 0, 0;
	 0, 0, (RXa-1.), 0, 0, 0;
	 0, -RYb, 0, RYc-4.0/3.0, 0, 0;
	 RYb, 0, 0, 0, RYc-4.0/3.0, 0;
	 0, 0, 0, 0, 0, RXc-4.0/3.0]; 
 %%%%%%%%%%%%%
 
mbXa = interp1(Rmbw(:,1), Rmbw(:,2), h(j), 'pchip'); 
mbYa = interp1(Rmbw(:,1), Rmbw(:,3), h(j), 'pchip'); 
mbYb = interp1(Rmbw(:,1), Rmbw(:,4), h(j), 'pchip'); 
mbXc = interp1(Rmbw(:,1), Rmbw(:,5), h(j), 'pchip'); 
mbYc = interp1(Rmbw(:,1), Rmbw(:,6), h(j), 'pchip'); 

Rmb = [(mbYa-1.), 0, 0, 0, mbYb, 0;
	 0, (mbYa-1.), 0, -mbYb, 0, 0;
	 0, 0, (mbXa-1.), 0, 0, 0;
	 0, -mbYb, 0, mbYc-4.0/3.0, 0, 0;
	 mbYb, 0, 0, 0, mbYc-4.0/3.0, 0;
	 0, 0, 0, 0, 0, mbXc-4.0/3.0];
 
 Re10 = Rsup10-Rmb;
 meig10(j) = min(eigs(Re10));

 Re2 = Rsup2-Rmb;
 meig2(j) = min(eigs(Re2));
 
 

 if((abs(h(j) -Rmbw(j,1)) < 1e-8) && first_h)
     disp(h(j)) 
     first_h = 0;
 end
 
 
end

figure(3)
plot(h,meig10)
hold all
plot(h,meig2)
hold off


epsilon = h-1;

Xa_asym = 1.0./epsilon - (1.0/5.0)*log(epsilon) + 0.971280;
Ya_asym = -(8.0/15.0)*log(epsilon) + 0.9588;
Yb_asym = -(-(1.0/10.0)*log(epsilon)-0.1895) - 0.4576*epsilon;
Yb_asym = Yb_asym*4./3.;
Xc_asym = 1.2020569 - 3.0*(pi*pi/6.0-1.0)*epsilon;
Xc_asym = Xc_asym*4./3.;
Yc_asym = -2.0/5.0*log(epsilon) + 0.3817 + 1.4578*epsilon;
Yc_asym = Yc_asym*4./3.;

asym_co = [h Xa_asym Ya_asym Yb_asym Xc_asym Yc_asym];


 figure(1)
 cols = parula(5);
 for i = 2:6
     subplot(2,3,i-1)
     loglog(Ri10(:,1),Ri10(:,i),'color',cols(i-1,:))
     hold all
     loglog(Ri2(:,1),Ri2(:,i),':','color',cols(i-1,:))
     hold all
     loglog(asym_co(:,1),asym_co(:,i),'-','color',[1 0 0],'linewidth',1)
     xlim([1.0001 1.1])
 end
 
 figure(2)
 cols = parula(5);
 for i = 2:6
     loglog(h,abs(Ri10(:,i)-Ri2(:,i))/abs(Ri10(:,i)),'color',cols(i-1,:))
     hold all
 end
 
 
 