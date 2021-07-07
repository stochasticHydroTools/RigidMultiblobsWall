close all
clear all

set(0,'defaulttextinterpreter','latex')
set(0,'defaultAxesFontName', 'Times New Roman')
set(0,'DefaultAxesFontSize',22)
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
plot_blobs_3D = 1
save_vertex_file = 0;
save_movie = 0

% number of bodies CHANGE THIS BY HAND


N = 15
Mx = 10
Mz = 10
bods = N*Mx*Mz

single_blob = 1;


if single_blob == 1
     suffix = '_single_blob';
else
    suffix = '';
end

root_name = ['robot_arm_N_' num2str(N) '_Mx_' num2str(Mx) '_Mz_' num2str(Mz)];
filename_config = ['run_' root_name suffix '.' root_name]

D = dlmread([filename_config '.config']);

% 'done reading'
remove = 1:(bods+1):length(D);
E = D;
E(remove,:) = [];

pos = E(:,1:3);
quat = E(:,4:7);


% .vertex for body

if single_blob == 0
    % blob radius
    a = 0.41642068;
    filename_vertex = 'shell_N_12_Rg_0_7921_Rh_1'
else
    a = 1;
    filename_vertex = 'blob'
end


data_body = dlmread([filename_vertex '.vertex']);
body = data_body(2:end,:)';

% read const
filename_const = root_name

D = dlmread([filename_const '.const']);
Nconst = D(3,1);
part_const = D(4:end,2:3);
links = D(4:end,5:11);

if save_movie == 1
  folder_movie = ['Movie_' filename_const]
  if ~exist(folder_movie, 'dir')
       mkdir(folder_movie)
  end
end



% number of blobs
[trash blobs] = size(body);

% platonic blob
[x,y,z] = sphere(20);


% colors for bodys
colormap(spring)

% length of each step
stepl = 1;

% nuber of steps
Nstep = 5; 

% view angle
%view_ang = [-126 8];
view_ang = [(-122:(-25/100):-147)' (4:(17/100):21)'];
view_ang = [0 0];




% for j = 1:stepl:stepl*Nstep  % loop over time
for j = stepl*Nstep  % loop over time

    j
    
    pos_all_blobs = zeros(bods*blobs,3);

    
    fig = figure(1);
    hold on
    set(fig, 'position',[10 10 1000 900])
% 
     
    
    for bod = 1:bods % loop over bodies
       % position (U1) and orientation (Q1) of body 'bod' at time level 'j'
       U1 = pos(bod + (j-1)*bods,:);
       Q1 = quat(bod + (j-1)*bods,:);
       
       % from rotation matrix
       R1 = Rot_From_Q(Q1(1),Q1(2:end));
       

       % configuation of body 'bod' at time level 'j'
       b1_pos = repmat(U1',1,blobs) + R1*body;
       

       pos_all_blobs((bod-1)*blobs+1:bod*blobs,:) = b1_pos';
      

       if plot_blobs_3D == 1 
           for i = 1:blobs % loop over blobs in body
              c1 = b1_pos(:,i); % position of blob 'i' 

              %plot spheres of radius a and center c1 using 'z' (the z of a sphere centered at 0) as color data
              surface(a*x+c1(1),a*y+c1(2),a*z+c1(3), ...  
                        'FaceLighting','phong', ... % a knob to turn for the lighting
                        'AmbientStrength',0.3, ... % a knob to turn for the lighting
                        'DiffuseStrength',0.6, ... % a knob to turn for the lighting
                        'Clipping','off',... % don't turn this on
                        'BackFaceLighting','lit', ... 
                        'SpecularStrength',0.0, ... % a knob to turn for the lighting
                        'SpecularColorReflectance',0.1, ... % a knob to turn for the lighting
                        'SpecularExponent',10, ... % a knob to turn for the lighting
                        'edgecolor','none') 


           end
       else
           
           plot3(b1_pos(1,:),b1_pos(2,:),b1_pos(3,:),'.')
       end

    end
    
    
    lighting gouraud


   % aspect ratio (probs dont change)
   daspect([1 1 1])
   % view angle
   view(view_ang)
   
   set(gca,'zminortick','on')
   set(gca,'yminortick','on')
   set(gca,'xminortick','on')

   set(gca,'ticklength',3*get(gca,'ticklength'))
   set(gca,'layer','top')
   set(fig,'color','w')
   

   xmin = min(pos_all_blobs(:,1));
   xmax = max(pos_all_blobs(:,1));
   
   zmin = min(pos_all_blobs(:,3));
   zmax = max(pos_all_blobs(:,3));

   xlim([xmin-2 xmax+2])
   zlim([zmin-2 zmax+2])
   pause(0.1)
   pause
   
   
   

    if save_movie == 1
        %save numbered pngs
       saveas(gcf,[folder_movie '/' filename_const '_' num2str(floor(j/stepl)+1) '.png'])
    end

   if j<Nstep
     clf
   end
   
  
end
