clc
close all

% a = 0.5;
% N = 39;
% x = (0:2*a:2*a*N)';
% y = 0*x; %0.25*randn(length(x),1);
% z = a+0.2+0*x; %0.25*randn(length(x),1);
% 
% plot(x,y)
% 
% configs_file = './chain_40.clones';
% dlmwrite(configs_file,length(x),'delimiter','\t','precision',5)
% dlmwrite(configs_file,[x y z 0*z+1 0*z 0*z 0*z],'-append','delimiter','\t','precision',12)

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%%%%%%%% Hairpin
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

a = 0.5;
%N = 28;
N = 25;
x = (2.1*a:2.1*a:2.1*a*N)';
r = 3*2.1*a;
% r = 1*2.1*a;
theta = linspace(pi/2,3*pi/2,10);
% theta = linspace(pi/2,3*pi/2,4);
cx = r*cos(theta');
cy = r*sin(theta');
xp = [flipud(x); cx;  x];
yp = [0*x-r; flipud(cy); 0*x+r];
zp = a+0.2+0*xp;

for k = 1:length(xp)
    if(k < length(xp))
        dist = sqrt((xp(k+1)-xp(k)).^2 + (yp(k+1)-yp(k)).^2);
        disp(dist)
    end
    circle2(xp(k),yp(k),a);
    hold all
end

plot(xp,yp,'o')
daspect([1 1 1])

Xps = [xp yp zp];
Qs = [];
for k = 1:length(xp)-1
    rhat = Xps(k+1,:)-Xps(k,:);
    rhat = rhat./norm(rhat);
    e_1 = [1 0 0];
    p = cross(e_1,rhat);
    s = dot(e_1,rhat) + norm(e_1)*norm(rhat);
    if(abs(dot(e_1,rhat)/norm(e_1)/norm(rhat) + 1) < 1e-8)
        q = [0 0 0 1];
    else
        q = [s p];
        q = q./norm(q);
    end
    Rot = Rot_From_Q(q(1),q(2:end));
    disp(norm(rhat - (Rot*(e_1'))'))
    
    Qs = [Qs; q];
    
end
Qs = [Qs; q];

configs_file = './hairpin_60.clones';
dlmwrite(configs_file,length(xp),'delimiter','\t','precision',5)
dlmwrite(configs_file,[Xps Qs],'-append','delimiter','\t','precision',12)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%% S curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% a = 0.5;
% %N = 28;
% N = 12;
% x = (2.1*a:2.1*a:2.1*a*N)';
% r = 3*2.1*a;
% % r = 1*2.1*a;
% theta = linspace(pi/2,3*pi/2,10);
% % theta = linspace(pi/2,3*pi/2,4);
% cx = r*cos(theta');
% cy = r*sin(theta');
% xp = [flipud(x); cx;  x; max(x)-cx+2.1*a; flipud(x)];
% xp = [max(x)+2.1*a; xp; 0];
% yp = [0*x-r; flipud(cy); 0*x+r; flipud(cy)+2*r; 0*x+3*r];
% yp = [yp(1); yp; yp(end)];
% zp = a+0.2+0*xp;
% 
% cols = jet(60);
% for k = 1:length(xp)
%     if(k < length(xp))
%         dist = sqrt((xp(k+1)-xp(k)).^2 + (yp(k+1)-yp(k)).^2);
%         disp(dist)
%     end
%     h = circle2(xp(k),yp(k),a);
%     set(h,'facecolor',cols(k,:))
%     hold all
% end
% 
% plot(xp,yp,'ko')
% daspect([1 1 1])
% 
% Xps = [xp yp zp];
% Qs = [];
% for k = 1:length(xp)-1
%     rhat = Xps(k+1,:)-Xps(k,:);
%     rhat = rhat./norm(rhat);
%     e_1 = [1 0 0];
%     p = cross(e_1,rhat);
%     s = dot(e_1,rhat) + norm(e_1)*norm(rhat);
%     if(abs(dot(e_1,rhat)/norm(e_1)/norm(rhat) + 1) < 1e-8)
%         q = [0 0 0 1];
%     else
%         q = [s p];
%         q = q./norm(q);
%     end
%     Rot = Rot_From_Q(q(1),q(2:end));
%     disp(norm(rhat - (Rot*(e_1'))'))
%     
%     Qs = [Qs; q];
%     
% end
% Qs = [Qs; q];
% 
% configs_file = './S_curve_60.clones';
% dlmwrite(configs_file,length(xp),'delimiter','\t','precision',5)
% dlmwrite(configs_file,[Xps Qs],'-append','delimiter','\t','precision',12)