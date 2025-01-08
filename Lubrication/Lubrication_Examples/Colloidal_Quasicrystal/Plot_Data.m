set(0,'defaulttextInterpreter','latex')
set(0,'defaultAxesTickLabelInterpreter','latex'); 
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultLineLineWidth',3);
set(0,'defaultAxesFontSize',35)


clc
clf


a = 2.25;
[sx, sy, sz] = sphere(50);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Un-comment for sigma sim %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% vid_name = 'Rhombus_Simulation';
%%% the first 10s of dipole equilibriation %%%
% NAME ='suspension_rhombus_N_12_random';
%%% 15s of evolution to final sigma config %%%
% NAME ='suspension_rhombus_N_12_random_eq1';
%%% 14s of holding final sigma config %%%
% NAME ='suspension_rhombus_N_12_random_eq2';
% f_name = ['./data/' vid_name '.' NAME '.config'];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Un-comment for ring sim %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% vid_name = 'Ladder_Simulation';
% NAME ='suspension_ladder_N_6_random';
% f_name = ['./data/' vid_name '.' NAME '.config'];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Un-comment for square sim %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vid_name = 'Square_Simulation';
NAME ='suspension_diamond_N_4_random';
f_name = ['./data/' vid_name '.' NAME '.config'];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



L = 128.0;

A = dlmread(f_name);


n_bods = A(1,1); 
B_z_history = A(1:(n_bods+1):end,2);

A(1:(n_bods+1):end,:) = [];
N = length(A)/n_bods;
dt = 80*0.000125;
skip = 4*1;




[X, Y] = meshgrid([-L/2:0.5:L/2],[-L/2:0.5:L/2]);

show_triad = 1;
show_beads = 1;


k = 0;

Ntime = length(A)/n_bods;
for i = Ntime 
    i
    k = k+1;
    clf
    
        

    
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
    
    scaleax = 1.8;
    xlim(scaleax*[-10 10])
    ylim(scaleax*[-10 10])
    zlim([0 8])

    out_cfgs = A((i-1)*n_bods+1:i*n_bods,1:3);
    out_vs = 0*out_cfgs;

    for j = 1:length(x)
        if show_beads==1
        fcol = [0.0 0.85 0.99]; %0.3*[1 1 1];
        h = surface(x(j)+a*sx,y(j)+a*sy,z(j)+a*sz,'facecolor',fcol,'edgecolor','none');
        set(h,'FaceLighting','gouraud',...%'facealpha',0.2, ...
        'AmbientStrength',0.3, ...
        'DiffuseStrength',0.6, ... 
        'Clipping','off',...
        'BackFaceLighting','lit', ...
        'SpecularStrength',1, ...
        'SpecularColorReflectance',1, ...
        'SpecularExponent',7,'FaceAlpha',1)
        end
    
        daspect([1 1 1])
        grid on
        view([0 90]) %view([-20 35]) %view([-140 10])% 


        set(gca, 'linewidth',3)
        ax1 = gca;
        set(ax1,'XTick',get(ax1,'YTick'));
        hold all
        
        if show_triad==1
        R = Rot_From_Q(s(j),p(j,:));
        V = R*eye(3);
        v = V(:,3);
        tv = V(:,2);
        ev = V(:,1);
        % hA1 = mArrow3([x(j); y(j); z(j)],[x(j); y(j); z(j)]+1.3*a*v,'color',[0.8 0.8 0.8],'stemWidth',0.1*a);
        % hold all
        % hA2 = mArrow3([x(j); y(j); z(j)],[x(j); y(j); z(j)]+1.3*a*tv,'color',[0.8 0.8 0.8],'stemWidth',0.1*a);
        % hold all
        hA3 = mArrow3([x(j); y(j); z(j)+2*a]-1*a*ev,[x(j); y(j); z(j)+2*a]+1*a*ev,'color',[0.45 0.1 1.0],'stemWidth',0.1*a,'tipWidth',0.2*a);
        hA3.FaceLighting = 'phong';
        hA3.AmbientStrength = 0.7;
        hA3.DiffuseStrength = 1.0;
        hA3.SpecularStrength = 0.9;
        hA3.SpecularExponent = 5;
        hA3.BackFaceLighting = 'unlit';
        hold all
        out_vs(j,:) = ev;

        end
        
    end



    rr = 25;

    zz = 45;
    lcol = 0.3*[255,255,255]./255;
    Nlights = 5;
    for ith = 0:Nlights-1
        th = ith*(2*pi/Nlights);
        lpos = [rr*cos(th) rr*sin(th) zz];
        light('Position',lpos,'Style','local','color',lcol)
    end
  
    %camlight


    B_z = B_z_history(i);
    %title(['t = ' num2str((i-1)*dt) ', $$B_z = $$' num2str(B_z)])
    E_z = 2.034707874856431e5*B_z;
    title(['t = ' num2str((i-1)*dt) ' (s), $$E_z = $$' num2str(round(E_z,2)) ' $$V/m$$'])


    drawnow

    hold off
    
end

