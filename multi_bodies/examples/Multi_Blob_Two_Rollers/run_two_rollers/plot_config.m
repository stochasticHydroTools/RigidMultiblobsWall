clc
clf

set(0,'defaulttextInterpreter','latex')
set(0, 'defaultAxesTickLabelInterpreter','latex'); 
set(0, 'defaultLegendInterpreter','latex');
set(0, 'defaultLineLineWidth',3);
set(0,'defaultAxesFontSize',35)

%%% set to 1 to plot the multiblob model (slow)
%%% set to 0 to just plot spheres with orientation markers (fast)
plot_blobs = 1;

%%% weather to print results
print_pngs = 0;

if plot_blobs==1
    %%% change this to the nuber of blobs used in the simulation
    blobs_plot = 12
    switch blobs_plot
        case 12
            Body = dlmread('../../../Structures/shell_N_12_Rg_0_7921_Rh_1.vertex');
        case 42
            Body = dlmread('../../../Structures/shell_N_42_Rg_0_8913_Rh_1.vertex');
        case 162
            Body = dlmread('../../../Structures/shell_N_162_Rg_0_9497_Rh_1.vertex');
        case 642
            Body = dlmread('../../../Structures/shell_N_642_Rg_0_9767_Rh_1.vertex');
        case 2562
            Body = dlmread('../../../Structures/shell_N_2562_Rg_0_9888_Rh_1.vertex');
    end
    n_blobs = Body(1,1);
    a = Body(1,2)/2;
    Body = Body(2:end,:)';
    
    n_yellow_blobs = round(0.95*n_blobs/12);
    yellow_idx = knnsearch(Body',Body(:,1)','k',n_yellow_blobs);
else
    a = 1.0;
end


[sx, sy, sz] = sphere(20);
fig1=figure(1)


A = dlmread('./data.two_rollers.config');
n_bods = A(1,1); 
A(1:(n_bods+1):end,:) = [];


N = length(A)/n_bods;
%%% number of timesteps saved
n_save = 1;
dt = n_save*0.01;
%%% number of timesteps to skip in visualization
skip = 1;


k = 0;
Lp=40;
[X, Y] = meshgrid([-Lp:0.5:Lp],[-Lp:0.5:Lp]);

set(fig1,'units','normalized','outerposition',[0 0 1 1])
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
        
        if plot_blobs==1
            R = Rot_From_Q(s(j),p(j,:));
            New_Body = repmat([x(j);y(j);z(j)],1,n_blobs) + R*Body;
            for m = 1:n_blobs
                bx = New_Body(1,m);
                by = New_Body(2,m);
                bz = New_Body(3,m);
                subplot(1,2,1)
                if sum(m == yellow_idx) == 1
                    fc = [1 1 0];
                else
                    fc = col;
                end
                    
                h = surface(bx+a*sx,by+a*sy,bz+a*sz,'facecolor',fc,'edgecolor','none');
                set(h,'FaceLighting','gouraud',...%'facealpha',0.2, ...
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
            end
        
        else
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
        end
        
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
    if print_pngs == 1
        print('-dpng',['../roller_pngs/rollers_' num2str(k) '.png'],'-r100')
    end
end
