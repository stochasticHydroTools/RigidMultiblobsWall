close all


% number of bodies CHANGE THIS BY HAND
bods = 256;

% read a file then remove unneeded bits

% D = dlmread('st1.boomerang_N_15.suspension.N.256.phi.0.25.file.101.config');
% 'done reading'
% remove = 1:(bods+1):length(D);
% E = D;
% E(remove,:) = [];


% .vertex for body
body = ...
[2.1  0.  0.
1.8  0.  0.
1.5  0.  0.
1.2  0.  0.
0.9  0.  0.
0.6  0.  0.
0.3  0.  0.
0.  0.  0.
0.  0.3  0.
0.  0.6  0.
0.  0.9  0.
0.  1.2  0.
0.  1.5  0.
0.  1.8  0.
0.  2.1  0.]';
% blob radius
a = 0.324557390919;

% number of blobs
[trash blobs] = size(body);

% platonic blob
[x,y,z] = sphere(20);
% periodic distance
Lp = 45.339607409;
% X,Y grid for wall plot
[X,Y] = meshgrid(-Lp/2:0.1:Lp/2,-Lp/2:0.1:Lp/2);
% colors for bodys
colormap(spring)

% length of each step
stepl = 20;

% nuber of steps
Nstep = 100; 

% view angle
%view_ang = [-126 8];
view_ang = [(-122:(-25/100):-147)' (4:(17/100):21)'];

for j = 1:stepl:stepl*Nstep  % loop over time
    clf
    fig = figure(1);
    set(fig, 'Units', 'Normalized', 'Outerposition', [0, 0, 1, 1]);
    % set up axes
    ax = axes('xlim',[-Lp/2 Lp/2],'ylim',[-Lp/2 Lp/2],'zlim',[-Lp/2 Lp/2],...
         'Visible','off','view',view_ang(floor(j/stepl)+1,:), ...
        'parent',fig);
    tic
    for bod = 1:bods % loop over bodies
       disp(bod)
       % position (U1) and orientation (Q1) of body 'bod' at time level 'j'
       U1 = E(bod + (j-1)*bods,1:3);
       Q1 = E(bod + (j-1)*bods,4:end);
       
       % from rotation matrix
       R1 = Rot_From_Q(Q1(1),Q1(2:end));
        
       % configuation of body 'bod' at time level 'j'
       b1_pos = repmat(U1',1,blobs) + R1*body;

       for i = 1:blobs % loop over blobs in body
          c1 = b1_pos(:,i); % position of blob 'i' 
          c1(1:2) = c1(1:2) - sign(c1(1:2)).*(abs(c1(1:2)) > Lp/2).*(Lp); % periodic adjustment
          
          %plot spheres of radius a and center c1 using 'z' (the z of a sphere centered at 0) as color data
          surface(a*x+c1(1),a*y+c1(2),a*z+c1(3),z, ...  
                    'FaceLighting','phong', ... % a knob to turn for the lighting
                    'AmbientStrength',0.3, ... % a knob to turn for the lighting
                    'DiffuseStrength',0.6, ... % a knob to turn for the lighting
                    'Clipping','off',... % don't turn this on
                    'BackFaceLighting','lit', ... 
                    'SpecularStrength',0.0, ... % a knob to turn for the lighting
                    'SpecularColorReflectance',0.1, ... % a knob to turn for the lighting
                    'SpecularExponent',10, ... % a knob to turn for the lighting
                    'edgecolor','none', ... %removes the mesh
                    'parent',ax) % asigns this plot to the graphics object ax
                
          % first light source       
          l1 = light('Position',[Lp/2 -Lp/2 10], ... % position 
                'Style','local', ...
                'Color',[1 0.4 0.1], ... % in the form [r g b] where 0 < r,g,b < 1 (black is [0 0 0])
                'parent',ax); % asigns this plot to the graphics object ax
            
          % second light source
          l2 = light('Position',[-(Lp/2) (Lp/2) 10], ... % position 
                'Style','local', ...
                'Color',[0.32 0.32 0.32], ... % color curently set to soft white
                'parent',ax); % asigns this plot to the graphics object ax
          hold all
       end
    end
   %plot wall at z = 0 with vairiable transucency (alphadata) for looks
   surf(X,Y,zeros(size(X)),'FaceAlpha','flat',...
    'AlphaDataMapping','scaled',...
    'AlphaData',sqrt(0.1*(X+15).^2 + 0.1*(Y-15).^2),...
    'FaceColor','k','edgecolor','none','Clipping','off', ...
    'parent',ax)
   lighting gouraud
   hold off
   % aspect ratio (probs dont change)
   daspect([1 1 1])
   % view angle
   view(view_ang(floor(j/stepl)+1,:))
   toc
   % zoom
   zoom(1.9)
   
   drawnow
   % save numbered pngs
   saveas(gcf,['BoomPngs/booms' num2str(floor(j/stepl)+1) '.png'])
   
  
end