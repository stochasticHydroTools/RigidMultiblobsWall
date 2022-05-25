
cfg = dlmread('../../multi_bodies/Structures/Cylinder_N_86_Lg_1_9384_Rg_0_1484.vertex');
cfg(1,:) = [];
cfg(cfg>1e4) = 0;

A = dlmread(['./data/DP_run.config']);
n_bods = round(A(1,1));
rem = mod(length(A),n_bods+1);
A(end-rem+1:end,:) = [];
A(1:n_bods+1:end,:) = [];


zmax = 1.0
a = 0.07419999999999999
L = 16.18021593796416 %40.106052394096004

[sx,sy,sz] = sphere(20);
[X, Y] = meshgrid([0:0.5:L],[0:0.5:L]);
k = 0;

skip = 10
plot_walls=0

for i = (length(A)/n_bods) %1:skip:
    clf
    i
    
    k = k+1;
    x = A((i-1)*n_bods+1:i*n_bods,1);
    y = A((i-1)*n_bods+1:i*n_bods,2);
    z = A((i-1)*n_bods+1:i*n_bods,3);
    s = A((i-1)*n_bods+1:i*n_bods,4);
    p = A((i-1)*n_bods+1:i*n_bods,5:end);
    
    
    r_vecs = [];
    for j = 1:length(x)
        R = Rot_From_Q(s(j),p(j,:));
        Bod = R*cfg' + [x(j);y(j);z(j)]*ones(1,length(cfg));
        r_vecs = [r_vecs Bod];
        for k = 1:length(Bod)
            xyz = Bod(:,k);
            h = surface(xyz(1)+a*sx,xyz(2)+a*sy,xyz(3)+a*sz,'facecolor','r','edgecolor','none');
            set(h,'FaceLighting','gouraud',...%'facealpha',0.2, ...
            'AmbientStrength',0.5, ...
            'DiffuseStrength',0.3, ... 
            'Clipping','off',...
            'BackFaceLighting','lit', ...
            'SpecularStrength',0.3, ...
            'SpecularColorReflectance',0.7, ...
            'SpecularExponent',1)
            hold all
        end
     end
        
        
    
        daspect([1 1 1])
        view([20 7]) %view([-20 35]) %view([-140 10])% 
        xlim([0 L])
        ylim([0 L])
        %zlim([-a 4*a+a])
        hold all
        

    if plot_walls
        surface(X,Y,0*X,'facecolor','k','edgecolor','none')
        hold all
        surface(X,Y,0*X+zmax,'facecolor','k','edgecolor','none','facealpha',0.1)
        hold all
    end
    
    camlight
    drawnow
    
    hold off
end

function R = Rot_From_Q(s,p)
    P = [0, -1*p(3), p(2)
        p(3), 0, -1*p(1)
        -1*p(2), p(1), 0];
    R = 2*((p'*p) + (s^2-0.5)*eye(3) + s*P);
end