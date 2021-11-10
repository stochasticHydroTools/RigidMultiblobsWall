clc
clf


a = 2.25;
[sx, sy, sz] = sphere(20);
figure(1)

% change f_name to your desired config file
% check cone angles for dimer
% f_name = './data/test_BD_1xB_54deg.suspension_35_eq.config';
% f_name = './data/test_BD_1xB_54.5deg.suspension_15_eq.config';

%f_name = './data/test_BD_1xB_56.5deg.suspension_71_L_150_eq2.config';
%f_name = './data/test_BD_1xB_56.5deg.suspension_71_L_150_eq3.config';
f_name = './data/test_BD_1xB_55.5deg.suspension_2.config';
A = dlmread(f_name);


n_bods = A(1,1); 


A(1:(n_bods+1):end,:) = [];
N = length(A)/n_bods;
dt = 20*0.001;
skip = 4*1;



L = 150;
[X, Y] = meshgrid([-L/2:0.5:L/2],[-L/2:0.5:L/2]);

show_triad = 1;
show_beads = 1;


%i = 50 72 86 98 154 190 250
k = 0;
i = 0;
%while i < (length(A)/n_bods) 
for i = 1:skip:(length(A)/n_bods)
    clf
    i
%     i = i + skip
%     
%     if i > 900
%         skip = 2;
%     end
    
    k = k+1;
    x = A((i-1)*n_bods+1:i*n_bods,1);
    y = A((i-1)*n_bods+1:i*n_bods,2);
    z = A((i-1)*n_bods+1:i*n_bods,3);
    s = A((i-1)*n_bods+1:i*n_bods,4);
    p = A((i-1)*n_bods+1:i*n_bods,5:end);
    
    for d = 1:length(x)
    while x(d) > L/2
        x(d) = x(d) - L;
    end
    while x(d) < -L/2
        x(d) = x(d) + L;
    end
    while y(d) > L/2
        y(d) = y(d) - L;
    end
    while y(d) < -L/2 
        y(d) = y(d) + L;
    end
    end
    
    
    
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
        xlim([-L/2 L/2])
        ylim([-L/2 L/2])
        zlim(a*[0 20])
        hold all
        
        if show_triad==1
        R = Rot_From_Q(s(j),p(j,:));
        V = R*eye(3);
        v = V(:,3);
        tv = V(:,2);
        ev = V(:,1);
        
        plot3([x(j),x(j)+1.2*a*v(1)],[y(j),y(j)+1.2*a*v(2)],[z(j),z(j)+1.2*a*v(3)],'m-^','linewidth',2','markersize',4)
        hold all
        plot3([x(j),x(j)+1.2*a*ev(1)],[y(j),y(j)+1.2*a*ev(2)],[z(j),z(j)+1.2*a*ev(3)],'c-^','linewidth',2','markersize',4)
        hold all
        plot3([x(j),x(j)+1.2*a*tv(1)],[y(j),y(j)+1.2*a*tv(2)],[z(j),z(j)+1.2*a*tv(3)],'y-^','linewidth',2','markersize',4)
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
    print('-dpng',['suspension_pngs/sus_' num2str(k) '.png'],'-r100')
end

% configs_file = './suspension_71_L_150_eq3.clones';
% dlmwrite(configs_file,length(x),'delimiter','\t','precision',5)
% dlmwrite(configs_file,A((i-1)*n_bods+1:i*n_bods,:),'-append','delimiter','\t','precision',12)
