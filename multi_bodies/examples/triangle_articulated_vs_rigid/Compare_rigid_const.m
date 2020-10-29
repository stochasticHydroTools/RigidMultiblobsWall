close all
clear all

set(0,'defaulttextinterpreter','latex')
set(0,'defaultAxesFontName', 'Times New Roman')
set(0,'DefaultAxesFontSize',22)
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');

% blob radius
a = 0.33;
mg = 1;
eta = 1;

Va = mg/(6*pi*eta*a);

Nstep = 50;
freq_save = 20;

dt = 0.1

t = dt*[0:Nstep-1]*freq_save;

%% Read constrained bodies
bods = 3;
filename_config = 'run_triangle_const_tol_1em5.triangle_N_3'
D = dlmread([filename_config '.config']);
remove = 1:(bods+1):length(D);
E = D;
E(remove,:) = [];
pos = E(:,1:3);
quat = E(:,4:7);

% .vertex for body
filename_vertex = 'shell_N_12_Rg_0_7921_Rh_1'
data_body = dlmread([filename_vertex '.vertex']);
body = data_body(2:end,:)';


% number of blobs
[trash blobs] = size(body);
pos_all_blobs_const = zeros(Nstep,bods*blobs,3);
for j = 1:Nstep
    for bod = 1:bods % loop over bodies
       % position (U1) and orientation (Q1) of body 'bod' at time level 'j'
       U1 = pos(bod + (j-1)*bods,:);
       Q1 = quat(bod + (j-1)*bods,:);
       
       % from rotation matrix
       R1 = Rot_From_Q(Q1(1),Q1(2:end));
       b1_pos = repmat(U1',1,blobs) + R1*body;
       pos_all_blobs_const(j,(bod-1)*blobs+1:bod*blobs,:) = b1_pos';
    end
end

%% Read rigid body
bods = 1;
filename_config = 'run_triangle_rigid.triangle_3_shells'
D = dlmread([filename_config '.config']);
remove = 1:(bods+1):length(D);
E = D;
E(remove,:) = [];
pos = E(:,1:3);
quat = E(:,4:7);

% .vertex for body
filename_vertex = 'triangle_3_shells'
data_body = dlmread([filename_vertex '.vertex']);
body = data_body(2:end,:)';


% number of blobs
[trash blobs] = size(body);
pos_all_blobs_rigid = zeros(Nstep,bods*blobs,3);
for j = 1:Nstep
    for bod = 1:bods % loop over bodies
       % position (U1) and orientation (Q1) of body 'bod' at time level 'j'
       U1 = pos(bod + (j-1)*bods,:);
       Q1 = quat(bod + (j-1)*bods,:);
       
       % from rotation matrix
       R1 = Rot_From_Q(Q1(1),Q1(2:end));
       b1_pos = repmat(U1',1,blobs) + R1*body;
       pos_all_blobs_rigid(j,(bod-1)*blobs+1:bod*blobs,:) = b1_pos';
    end
end


%% Compare data
Nblobs = bods*blobs;
diff_time = pos_all_blobs_rigid - pos_all_blobs_const;
dist_time = zeros(Nstep,Nblobs);
for j = 1:Nstep
    for n = 1:Nblobs
        dist_time(j,n) =  norm(squeeze(diff_time(j,n,:)));
    end
end
mean_dist_time = mean(dist_time,2);
figure
plot(t,mean_dist_time,'-')
xlabel('$t$')
ylabel('$\|\mathbf{x}_{blob}^{rigid} - \mathbf{x}_{blob}^{const}\|$')


mean_error_per_time_step = mean(diff(mean_dist_time)/freq_save)




    
