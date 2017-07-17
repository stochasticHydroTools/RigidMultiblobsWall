%Author: Zhe Chen

function [prob,Sigma]=prob_gen_hori(f,r_step,r_num,H)
%Get the prob density in parallel direction from the locations of all particles
%---------------------------------------------------------------------------
%Input:
%f:file directory which contains location of all the particles, vector:(N*2)
%r_step: r interval to sample on $ln(p)\sim r^2$
%r_num:The number of r\_step where we calculate the P(r).
%H: height of the plane
%---------------------------------------------------------------------------
%Output:
%prob: prob distribution of P(r), vector:1*r_num
%Sigma: covariance matrix , 2-by-2 matrix.
%---------------------------------------------------------------------------
loc=dlmread(f);
N=loc(1,1);
loc=loc(2:end,1:2);
Sigma=cov(loc);
if any(any(isnan(Sigma),1))
    disp('NaN found!')
end
loc=loc.^2;
loc=((sqrt(loc(:,1)+loc(:,2))));
prob=zeros(1,r_num);
for i=1:N
    index=floor(loc(i)/r_step)+1;
    if index<=r_num
        prob(index)=prob(index)+1;
    end
end
% for i=1:N
%     temp=(loc(i));
%     index=ceil(abs(loc(i))/r_step)-1;
%     if index<=r_num-1
%         if temp<0
%             index=r_num-index;
%         elseif temp>0
%             index=r_num+1+index;
%         else
%             index=0;
%         end
%         if index==0
%             continue;
%         end
%         prob(index)=prob(index)+1;
%     end
% end
prob=prob/(N*r_step^2*pi);
for i=1:r_num
    prob(i)=prob(i)/(2*i-1);
end
end