%A = dlmread(['data/Slit_channel.SC_0.4_L_128.7923.config']);
A = dlmread(['data/Slit_channel_6a.SC_0.4_L_128.7923.config']);

n_bods = round(A(1,1));
rem = mod(length(A),n_bods+1);
A(end-rem+1:end,:) = [];
A(1:n_bods+1:end,:) = [];


dt = 0.01;
a = 1.0155;
L = 128.7923;
[sx,sy,sz] = sphere(10);
[X, Y] = meshgrid([0:0.5:L],[0:0.5:L]);
k = 0;
disp('data read')
%%
gold = [252, 176, 69]./255;
red = [253, 29, 29]./255;
purp = 0.75*[119, 9, 193]./255;
blue = [0 0.5 1];
black = [0,0,0];
%teal = [72,209,204]./255;
teal = [52,229,224]./255;
fuscia = [217,2,125]./255;

% skip = 5
% start_c = 42;
% count = 0;
% t_old = 2.05;


% skip = 5
% start_c = 298;
% count = 0;
% t_old = 14.84;

skip = 5
start_c = 0;
count = 0;
t_old = 0.0;

%p_hat = zeros(3,n_bods);

Max_H = 7.5923;

tt = linspace(0,1,256)';
ccmap = tt*fuscia + (1-tt)*gold;


for i = (length(A)/n_bods) %1:skip:  %
    clf
    i
%     i = i + skip
%     
%     if i > 900
%         skip = 2;
%     end
    
    k = k+1;
    s = A((i-1)*n_bods+1:i*n_bods,4);
    p = A((i-1)*n_bods+1:i*n_bods,5:end);
    
    xyz = A((i-1)*n_bods+1:i*n_bods,1:3);

                
    
    
    for j = 1:n_bods

        for dd = 1:2
            while xyz(j,dd) < 0
                xyz(j,dd) = xyz(j,dd) + L;
            end
            while xyz(j,dd) > L
                xyz(j,dd) = xyz(j,dd) - L;
            end
        end
        
        if mod(j,64) ~= 0
            t = (xyz(j,3)/a-1)/(Max_H/a-1);
            H_col = t*fuscia + (1-t)*gold;
            h = circle2(xyz(j,1),xyz(j,2),a,black,H_col);
        else
            h = circle2(xyz(j,1),xyz(j,2),a,black,teal);
        end
        %h = circle2(xyz(j,1),xyz(j,2),a,black,blue);

        hold all
        R = Rot_From_Q(s(j),p(j,:));
        dot = R(:,3);
        %p_hat(:,j) = dot;
        if(dot(3) > 0)
           plot(xyz(j,1)+a*dot(1),xyz(j,2)+a*dot(2),'.','color',black,'markersize',12)
        end
        daspect([1 1 1])
        xlim([0 L])
        ylim([0 L])
        %zlim([-a 4*a+a])
        hold all
        

    end


    title(['t = ' num2str(((i-1)*dt)+t_old)])
    drawnow
    
    hold off
    count = count+1;
    disp('not saving image')
    %print('-dpng',['Channel_Pngs/frame_' num2str(start_c+count) '.png'],'-r100')
end
colormap(ccmap)
cbar = colorbar;
caxis([1 Max_H/a])
cbar.LineWidth=3;
cbar.TickLabelInterpreter='Latex';
cbar.Label.Interpreter = 'Latex';
cbar.Label.String = '$$h/a$$';
cbar.Label.FontSize = 25;

% configs_file = './SC_0.4_L_128.7923_eq1.clones';
% dlmwrite(configs_file,length(x),'delimiter','\t','precision',5)
% dlmwrite(configs_file,[xyz s p],'-append','delimiter','\t','precision',12)


function R = Rot_From_Q(s,p)
    P = [0, -1*p(3), p(2)
        p(3), 0, -1*p(1)
        -1*p(2), p(1), 0];
    R = 2*((p'*p) + (s^2-0.5)*eye(3) + s*P);
end


function h = circle2(x,y,r,col,fcol)
d = r*2;
px = x-r;
py = y-r;
h = rectangle('Position',[px py d d],'Curvature',[1,1],...
              'EdgeColor',col,'FaceColor',fcol,'linewidth',1.5);
daspect([1,1,1])
end