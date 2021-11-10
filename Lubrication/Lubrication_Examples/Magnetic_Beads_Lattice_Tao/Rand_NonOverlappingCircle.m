close all
clear all



r = 2.25;

L = 150;
phi = 0.05;
nCircles = round(L^2*phi/(pi*r^2));

circles = zeros(nCircles ,2);

disp(['phi = ' num2str(pi*r^2*nCircles/L/L)])
disp(['N = ' num2str(nCircles)])


for i=1:nCircles
    %Flag which holds true whenever a new circle was found
    newCircleFound = false;
    
    %loop iteration which runs until finding a circle which doesnt intersect with previous ones
    while ~newCircleFound
        x = -L/2.0 + (L)*rand(1);
        y = -L/2.0 + (L)*rand(1);
        
        %calculates distances from previous drawn circles
        prevCirclesY = circles(1:i-1,2);
        prevCirclesX = circles(1:i-1,1);
        
        DX = abs(prevCirclesX-x);
        DY = abs(prevCirclesY-y);
        
        DX = min(DX,L - DX);
        DY = min(DY,L - DY);
        
        distFromPrevCircles = ((DX).^2+(DY).^2).^0.5;
                
        
        %if the distance is not to small - adds the new circle to the list
        if i==1 || sum(distFromPrevCircles<=2*r)==0
            newCircleFound = true;
            circles(i,:) = [x y];
            circle2(x,y,r,[0 0 0]);
            axis([-L/2.0 L/2.0 -L/2.0 L/2.0]);
            daspect([1 1 1])
        end
    
    end
    hold all
end

for m = 1:length(circles)
    flag = 0;
    xr = circles(m,1);
    yr = circles(m,2);
    if circles(m,1) > (L/2.0-2*r)
        xr = circles(m,1) - L;
        flag = 1;
    end
    if circles(m,1) < (-L/2.0+2*r)
        xr = circles(m,1) + L;
        flag = 1;
    end
    if circles(m,2) > (L/2.0-2*r)
        yr = circles(m,2) - L;
        flag = 1;
    end
    if circles(m,2) < (-L/2.0+2*r)
        yr = circles(m,2) + L;
        flag = 1;
    end
    
    if flag
        hold all
        circle2(xr,yr,r,[1 0 0]);
    end
end
        


pos = [circles (r+0.1)+0*circles(:,1)];
pos_q = [pos 1.0+0*pos(:,1) 0*pos];
configs_file = ['./suspension_' ...
                num2str(nCircles) '_L_' num2str(L) '.clones'];
dlmwrite(configs_file,length(pos_q),'delimiter','\t','precision',5)
dlmwrite(configs_file,pos_q,'-append','delimiter','\t','precision',12)


function h = circle2(x,y,r,col)
d = r*2;
px = x-r;
py = y-r;
h = rectangle('Position',[px py d d],'Curvature',[1,1],'EdgeColor',col);
daspect([1,1,1])
end
