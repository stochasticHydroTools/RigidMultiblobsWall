clc
clf

set(0,'defaulttextInterpreter','latex')
set(0, 'defaultAxesTickLabelInterpreter','latex'); 
set(0, 'defaultLegendInterpreter','latex');
set(0, 'defaultLineLineWidth',3);
set(0,'defaultAxesFontSize',35)


a = 1.0;
[sx, sy, sz] = sphere(20);
figure(1)


A = dlmread('./data.two_rollers.config');
n_bods = A(1,1); 
A(1:(n_bods+1):end,:) = [];


N = length(A)/n_bods;
n_save = 2;
dt = n_save*0.01;
skip = 10;


k = 0;
Lp=40
[X, Y] = meshgrid([-Lp:0.5:Lp],[-Lp:0.5:Lp]);


for i = 1:skip:(length(A)/n_bods)
    clf
    i
    k = k+1;
    x = A((i-1)*n_bods+1:i*n_bods,1);
    y = A((i-1)*n_bods+1:i*n_bods,2);    
    z = A((i-1)*n_bods+1:i*n_bods,3);
    s = A((i-1)*n_bods+1:i*n_bods,4);
    p = A((i-1)*n_bods+1:i*n_bods,5:end);

    
    for j = 1:length(x)
        if j > 1
            col = [255,20,147]/255;
        else
            col = [0 0.5 1];
        end
        a_scale = 0.4;
        R = Rot_From_Q(s(j),p(j,:));
        e_z = R*[0;0;a-0.5*a_scale];
        subplot(1,2,1)
        h = surface(x(j)+a*sx,y(j)+a*sy,z(j)+a*sz,'facecolor',col,'edgecolor','none');
        set(h,'FaceLighting','gouraud',...%'facealpha',0.2, ...
        'AmbientStrength',0.5, ...
        'DiffuseStrength',0.3, ... 
        'Clipping','off',...
        'BackFaceLighting','lit', ...
        'SpecularStrength',0.3, ...
        'SpecularColorReflectance',0.7, ...
        'SpecularExponent',1)
    
        hold all 
        hb = surface(x(j)+e_z(1)+a_scale*a*sx,y(j)+e_z(2)+a_scale*a*sy,z(j)+e_z(3)+a_scale*a*sz,'facecolor',[1 1 0],'edgecolor','none');
        set(hb,'FaceLighting','gouraud',...%'facealpha',0.2, ...
        'AmbientStrength',0.5, ...
        'DiffuseStrength',0.3, ... 
        'Clipping','off',...
        'BackFaceLighting','lit', ...
        'SpecularStrength',0.3, ...
        'SpecularColorReflectance',0.7, ...
        'SpecularExponent',1)
     
        daspect([1 1 1])
        view([-20 35]) %view([0 90]) %view([-140 10])% 
        axis([mean(x)-5 mean(x)+5 mean(y)-5 mean(y)+5 0 10])
        xlabel('x')
        ylabel('y')
        zlabel('z')
        hold all
        
        subplot(1,2,2)
        xs = A(j:n_bods:i,1);
        ys = A(j:n_bods:i,2);    
        zs = A(j:n_bods:i,3);
        plot(xs,zs,'color',col)
        hold all
        subplot(1,2,2)
        plot(A(1:n_bods,1),A(1:n_bods,3),'o','markerfacecolor','k','color','k','markersize',8)
        hold all
        %axis([min(A(1:n_bods*i,1)) max(A(1:n_bods*i,1)) min(A(:,3)) max(A(:,3))])
        axis tight
        xlabel('x')
        ylabel('z')
        hold all
    end
    subplot(1,2,1)
    [X, Y] = meshgrid([mean(x)-20:1:mean(x)+20],[mean(y)-20:1:mean(y)+20]);
    surface(X,Y,0*X,'facecolor','k','edgecolor','none')
    l1 = light('Position',[mean(x)+20 mean(y)+20 30], ...
    'Style','local', ...
    'Color',1*[1 1 1]); 
    title(['t = ' num2str((i-1)*dt)])
    drawnow
    
    hold off
    %print('-dpng',['chain_pngs/helix_40_no_twist_' num2str(k) '.png'],'-r100')
end
