set(0,'defaulttextInterpreter','latex')
set(0,'defaultAxesTickLabelInterpreter','latex'); 
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultLineLineWidth',3);
set(0,'defaultAxesFontSize',35)


clc
clf


a = 2.25;
[sx, sy, sz] = sphere(50);



step_size = 0.1; 
step_Len = 5;
B_z_s = 0.535;
vid_name = 'WithInduced_B0_0p92_B_z_0p535_step_0p0_5_10_5s_max_0p67_perm_8p6aJ_per_mT_mutual_dipole_no_mut_E_or_B'
NAME ='ladder_N_6';
f_name = ['./data/' vid_name '.suspension_' NAME '_random.config'];


L = 128.0;

A = dlmread(f_name);


n_bods = A(1,1); 


A(1:(n_bods+1):end,:) = [];
N = length(A)/n_bods;
dt = 80*0.000125;
skip = 4*1;




[X, Y] = meshgrid([-L/2:0.5:L/2],[-L/2:0.5:L/2]);

show_triad = 1;
show_beads = 1;


%i = 50 72 86 98 154 190 250
k = 0;
i = 0;
%while i < (length(A)/n_bods)
Ntime = length(A)/n_bods;
for i = Ntime %10*376 % %Ntime %2500
    i
    k = k+1;
    %subplot(1,2,k)
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
        fcol = 0.3*[1 1 1];
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
        hA1 = mArrow3([x(j); y(j); z(j)],[x(j); y(j); z(j)]+1.3*a*v,'color',[0.8 0.8 0.8],'stemWidth',0.1*a);
        hold all
        hA2 = mArrow3([x(j); y(j); z(j)],[x(j); y(j); z(j)]+1.3*a*tv,'color',[0.8 0.8 0.8],'stemWidth',0.1*a);
        hold all
        hA3 = mArrow3([x(j); y(j); z(j)],[x(j); y(j); z(j)]+1.4*a*ev,'color','m','stemWidth',0.2*a);
        hold all
        out_vs(j,:) = ev;

        end
        
    end
    %surface(X,Y,0*X,'facecolor','k','edgecolor','none')
    %l1 = light('Position',[15 15 max(z)+100], ...
    %'Style','local', ...
    %'Color',1*[1 1 1]);

    tt = (i-1)*dt;
    theta = 2*pi*80*tt;
    apos = [-12.5; 12.5; a];
    %B_I = @(t) [2.0*abs(cos(t))-1.0; sin(t); 0.0*t];
    B_I_abs = @(t) [2.0*abs(cos(t))-(8/(2*pi)); sin(t); 0.0*t];
    B_I = @(t) [cos(2.0*t); sin(t); 0.0*t];
    ev = B_I(theta);
    fA = mArrow3(apos,apos+2.5*ev,'color',[0 0.7 1],'tipWidth',0.3);
    apos = [12.5; 12.5; a];
    ev = B_I_abs(theta);
    %fA = mArrow3(apos,apos+2.5*ev,'color',[1 0.7 0],'tipWidth',0.3);
    thg = 0:0.01:2*pi;
    apos = [-12.5; 12.5; a];
    evs = B_I(thg);
    plc = apos + 2.5*evs;
    plot3(plc(1,:),plc(2,:),plc(3,:),'color',[0.1 0.6 1.0])
    apos = [12.5; 12.5; a];
    evs = B_I_abs(thg);
    plc = apos + 2.5*evs;
    %plot3(plc(1,:),plc(2,:),plc(3,:),'color',[1.0 0.6 0.1])



    camlight
    % BTC = floor((i-1)*dt/3.0);
    % B_z = (2.0*BTC)/40.25;
    % BTC = floor((i-1)*dt/5.0);
    % B_z = 0.5+BTC*0.005; %0.4
    %B_z = 0.0;
    %B_z = 0.25+BTC*0.005; %0.25 for Yan_ladder
    
    BTC = floor((i-1)*dt/step_Len);
    B_z = B_z_s+step_size*BTC; %0.4
    %B_z = min(B_z,0.4);
    title(['t = ' num2str((i-1)*dt) ', $$B_z = $$' num2str(B_z)])
    
    
    % hold off
    % hs(2) = copyobj( hs(1) , hf ) ;
    % set(hs(2),'OuterPosition',axpos{1})
    % title(hs(1),[' '])
    % view(hs(2), [0 0])
    % xlim(hs(2),scaleax*[-12 12])
    % %camlight
    % drawnow

    drawnow

    hold off
    
    if(false)
    newSubFolder = ['./Cfgs/' NAME '_' vid_name];
    if ~exist(newSubFolder, 'dir')
        mkdir(newSubFolder);
    end
    cfg_f_nm = [newSubFolder '/cfg_' num2str(k) '.txt'];
    dlmwrite(cfg_f_nm,out_cfgs,'delimiter','\t','precision',12)
    vec_f_nm = [newSubFolder '/vec_' num2str(k) '.txt'];
    dlmwrite(vec_f_nm,out_vs,'delimiter','\t','precision',12)
    end
end

% posmod = A((i-1)*n_bods+1:i*n_bods,:);
% configs_file = './suspension_ladder_N_6_random_eq2.clones';
% dlmwrite(configs_file,length(x),'delimiter','\t','precision',5)
% dlmwrite(configs_file,posmod,'-append','delimiter','\t','precision',12)

