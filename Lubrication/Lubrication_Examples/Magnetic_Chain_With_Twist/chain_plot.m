clc
clf


a = 0.5;
[sx, sy, sz] = sphere(20);
figure(1)

% change f_name to your desired config file
f_name = './data/twist_helix.chain_60.config';
A = dlmread(f_name);


n_bods = A(1,1); 


A(1:(n_bods+1):end,:) = [];
N = length(A)/n_bods;
dt = 20*0.001;
skip = 4; %4*20;

Nhist = 100;
cols = jet(Nhist);
k = 0;

[X, Y] = meshgrid(a*[-30:0.5:160],a*[-30:0.5:160]);

show_triad = 1;
show_beads = 1;


%i = 50 72 86 98 154 190 250

for i = 50:skip:(length(A)/n_bods) %1:skip:(length(A)/n_bods)
    clf
    i
    k = k+1;
    x = A((i-1)*n_bods+1:i*n_bods,1);
    y = A((i-1)*n_bods+1:i*n_bods,2);
    z = A((i-1)*n_bods+1:i*n_bods,3);
    s = A((i-1)*n_bods+1:i*n_bods,4);
    p = A((i-1)*n_bods+1:i*n_bods,5:end);
    
    for j = 1:length(x)
        if show_beads==1
        h = surface(x(j)+a*sx,y(j)+a*sy,z(j)+a*sz,'facecolor','r','edgecolor','none');
        set(h,'FaceLighting','gouraud',...%'facealpha',0.2, ...
        'AmbientStrength',0.5, ...
        'DiffuseStrength',0.3, ... 
        'Clipping','off',...
        'BackFaceLighting','lit', ...
        'SpecularStrength',0.3, ...
        'SpecularColorReflectance',0.7, ...
        'SpecularExponent',1)
        end
    
        daspect([1 1 1])
        view([0 90]) %view([-20 35]) %view([-140 10])% 
        xlim(a*[-30 160])
        ylim(a*[-10 100])
        zlim(a*[0 20])
        hold all
        
        if show_triad==1
        R = Rot_From_Q(s(j),p(j,:));
        v = R*[0;0;1];
        if(j < length(x))
            tv = [x(j+1);y(j+1);z(j+1)]-[x(j);y(j);z(j)];
            tv = tv/norm(tv);
        else
            tv = [x(j);y(j);z(j)]-[x(j-1);y(j-1);z(j-1)];
            tv = tv/norm(tv);
        end
        ev = cross(tv,v);
        plot3([x(j),x(j)+v(1)],[y(j),y(j)+v(2)],[z(j),z(j)+v(3)],'m-^','linewidth',2','markersize',4)
        hold all
        plot3([x(j),x(j)+ev(1)],[y(j),y(j)+ev(2)],[z(j),z(j)+ev(3)],'c-^','linewidth',2','markersize',4)
        hold all
        plot3([x(j),x(j)+tv(1)],[y(j),y(j)+tv(2)],[z(j),z(j)+tv(3)],'g-^','linewidth',2','markersize',4)
        hold all
        end
    end
    surface(X,Y,0*X,'facecolor','k','edgecolor','none')
    l1 = light('Position',[15 15 max(z)+100], ...
    'Style','local', ...
    'Color',1*[1 1 1]); 
    title(['t = ' num2str((i-1)*dt)])
    drawnow
    
    hold off
    %print('-dpng',['chain_pngs/helix_' num2str(k) '.png'],'-r100')
end

% configs_file = './chain_60_eq.clones';
% dlmwrite(configs_file,length(x),'delimiter','\t','precision',5)
% dlmwrite(configs_file,A((i-1)*n_bods+1:i*n_bods,:),'-append','delimiter','\t','precision',12)
