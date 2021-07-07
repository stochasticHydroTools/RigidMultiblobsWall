close all
clear all

set(0,'defaulttextinterpreter','latex')
set(0,'defaultAxesFontName', 'Times New Roman')
set(0,'DefaultAxesFontSize',22)
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');


a = 1;
dist = 2.5*a;
Nbody_arm = 30;
Larm = (Nbody_arm -1)*dist;
Nxarms = 10;
Nzarms = 10;
Narms = Nxarms * Nzarms;
dxarm = 4*a;
dzarm = 4*a;

single_blob = 0;


Nbodies = Nbody_arm*Narms;

% place bodies
pos_bodies = zeros(Nbodies,3);
quat_bodies = zeros(Nbodies,4);
% all bodies aligned with x axis
quat_bodies(:,1) = 1;
for nx = 1:Nxarms
    for nz = 1:Nzarms
        ind_body = (nx-1)*Nzarms*Nbody_arm + (nz-1)*Nbody_arm;
        for m = 1:Nbody_arm            
            pos_bodies(ind_body + m,1) = (nx-1)*(Larm+dxarm) + (m-1)*dist;
            pos_bodies(ind_body + m,3) = (nz-1)*dzarm;
        end
    end
end



hfig = figure
hold on
box on
[xxs,yys,zzs] = sphere; 
plot3(pos_bodies(:,1),pos_bodies(:,2),pos_bodies(:,3),'ok','markerfacecolor','k')


alpha(.5)
set(gca,'zminortick','on')
set(gca,'yminortick','on')
set(gca,'xminortick','on')

axis equal
axis([min(pos_bodies(:,1))-a max(pos_bodies(:,1))+a -a a min(pos_bodies(:,3))-a max(pos_bodies(:,3))+a])

set(gca,'ticklength',3*get(gca,'ticklength'))
set(gca,'layer','top')
shading interp
view([0 0])



%% Generate clones files
filename = ['robot_arm_N_' num2str(Nbody_arm) '_Mx_' num2str(Nxarms) '_Mz_' num2str(Nxarms) ]
to_save = [Nbodies  zeros(1,6);pos_bodies quat_bodies ];
dlmwrite([filename '.clones'],to_save,'delimiter',' ','precision',16) 

%% Generate list vertex files
if single_blob == 1
    filename_vertex = 'blob.vertex';
    suffix = '_single_blob';
else
    filename_vertex = 'shell_N_12_Rg_0_7921_Rh_1.vertex'; 
    suffix = '';
end

fid= fopen([filename suffix '.list_vertex'],'w');
for n = 1:Nbody_arm
    fprintf(fid,[filename_vertex '\n']);
end

fclose(fid)

%% Generate constraint files for squares making the ribbon
Nconst_per_body = Nbody_arm - 1;
if single_blob == 1
    Nbead_per_body = 1;
else
    Nbead_per_body = 12;
end
fid= fopen([filename '.const'],'w');
fprintf(fid,'%i \n',Nbody_arm);
fprintf(fid,'%i ',Nbead_per_body*ones(1,Nbody_arm));
fprintf(fid,'\n');
fprintf(fid,'%i \n',Nconst_per_body);

nparam = 6;
constraints = zeros(Nconst_per_body,10);
constraints(:,4) = nparam;

% Constraints 
for n = 1:Nconst_per_body   
   part1 = n-1;
   constraints(n,2) = part1;
   part2 = n;
   constraints(n,3) = part2;   
   link1 = [dist/2 0 0]';
   constraints(n,5:7) = link1';   
   link2 = [-dist/2 0 0]';
   constraints(n,8:10) = link2';  
   fprintf(fid,'%i %i %i %i %f %f %f %f %f %f',constraints(n,:));
   fprintf(fid,'\n');
end

fclose(fid)





