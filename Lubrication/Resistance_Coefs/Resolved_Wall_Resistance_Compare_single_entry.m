% sup = [10.018295999999999  0.888180472982180  0.943989242719936  0.000008465684788  0.750365685562452  0.750225792564849];
sup = [10.018295999999999	0.888265249903143	0.944074315089753	0.000008449173193	0.750082434942972	0.749942535581558];
% sup = [2.49980000e+00 5.69471699e-01 7.78492438e-01 1.58931141e-03 7.44167966e-01 7.34981533e-01];
% sup = [1.000100000000000	0.014082680380347	0.242551088427938	0.095641456031245	0.623887876400550	0.282914080926557];

Rmbw = dlmread('res_scalars_wall_MB.txt');
Rmbw = flipud(Rmbw);

h = sup(1);

Xa = sup(2);
Ya = sup(3);
Yb = sup(4);
Xc = sup(5);
Yc = sup(6);

denom = Ya*Yc - Yb*Yb;
RXa = 1.0/Xa;
RYa = Yc/denom;
RYb = -Yb/denom;
RXc = 1.0/Xc;
RYc = Ya/denom; 

Rsup = [(RYa-1.), 0, 0, 0, RYb, 0;
	 0, (RYa-1.), 0, -RYb, 0, 0;
	 0, 0, (RXa-1.), 0, 0, 0;
	 0, -RYb, 0, RYc-4.0/3.0, 0, 0;
	 RYb, 0, 0, 0, RYc-4.0/3.0, 0;
	 0, 0, 0, 0, 0, RXc-4.0/3.0];
 
mbXa = interp1(Rmbw(:,1), Rmbw(:,2), h, 'pchip'); 
mbYa = interp1(Rmbw(:,1), Rmbw(:,3), h, 'pchip'); 
mbYb = interp1(Rmbw(:,1), Rmbw(:,4), h, 'pchip'); 
mbXc = interp1(Rmbw(:,1), Rmbw(:,5), h, 'pchip'); 
mbYc = interp1(Rmbw(:,1), Rmbw(:,6), h, 'pchip'); 

Rmb = [(mbYa-1.), 0, 0, 0, mbYb, 0;
	 0, (mbYa-1.), 0, -mbYb, 0, 0;
	 0, 0, (mbXa-1.), 0, 0, 0;
	 0, -mbYb, 0, mbYc-4.0/3.0, 0, 0;
	 mbYb, 0, 0, 0, mbYc-4.0/3.0, 0;
	 0, 0, 0, 0, 0, mbXc-4.0/3.0];
 
 R = Rsup-Rmb;
 min(eigs(R))
 

 
 
 