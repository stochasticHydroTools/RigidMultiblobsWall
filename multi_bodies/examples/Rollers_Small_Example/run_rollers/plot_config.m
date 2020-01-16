clc
clf


a = 0.656;
[sx, sy, sz] = sphere(20);
fig1=figure(1)


A = dlmread('./data.16_lines.config');
n_bods = A(1,1); 
A(1:(n_bods+1):end,:) = [];


N = length(A)/n_bods;
%%% the number of saved timesteps from the inputfile
n_save = 5;
dt = n_save*0.016;
%%% number of steps to skip in visualization
skip = 1;
%%% weather to print results
print_pngs = 1;


k = 0;
Lp=86.7232
[X, Y] = meshgrid([0:0.5:Lp],[0:0.5:Lp]);

%set(fig1,'units','normalized','outerposition',[0 0 0.5 0.5]) % Half a screen
set(fig1,'units','normalized','outerposition',[0 0 1 1]) % Full screen

for i = 1:skip:(length(A)/n_bods)
    clf
    i
    k = k+1;
    x = A((i-1)*n_bods+1:i*n_bods,1);
    
    y = A((i-1)*n_bods+1:i*n_bods,2);
    y = rem(y,Lp) - 0.5*Lp*sign(y)+0.5*Lp;
    
    z = A((i-1)*n_bods+1:i*n_bods,3);

    
    x_exts = [min(x)-5 max(x)*1.2];
    
    for j = 1:length(x)
        if j > length(x)-63
            col = [1 0.5 0];
        else
            col = [0 0.5 1];
        end
        h = surface(x(j)+a*sx,y(j)+a*sy,z(j)+a*sz,'facecolor',col,'edgecolor','none');
        set(h,'FaceLighting','gouraud',...%'facealpha',0.2, ...
        'AmbientStrength',0.5, ...
        'DiffuseStrength',0.3, ... 
        'Clipping','off',...
        'BackFaceLighting','lit', ...
        'SpecularStrength',0.3, ...
        'SpecularColorReflectance',0.7, ...
        'SpecularExponent',1)
    
        daspect([1 1 1])
        %view([0 90]) %view([-20 35]) %view([-140 10])% 
        xlim(x_exts)
        ylim([0 Lp])
        zlim([0 20])
        hold all
    end
    [X, Y] = meshgrid([x_exts(1):1:x_exts(2)],[0:1:Lp]);
    surface(X,Y,0*X,'facecolor','k','edgecolor','none')
    l1 = light('Position',[15 15 max(z)+100], ...
    'Style','local', ...
    'Color',1*[1 1 1]); 
    title(['t = ' num2str((i-1)*dt)])
    drawnow
    
    hold off

    if print_pngs == 1
        print('-dpng',['../roller_pngs/rollers_' num2str(k) '.png'],'-r100')
    end

end
