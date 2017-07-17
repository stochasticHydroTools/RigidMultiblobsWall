%Author: Zhe Chen

function prob=prob_gen(loc,r_step,r_num)
%Get the prob density from the locations of all particles
%---------------------------------------------------------------------------
%Input:
%loc: location of all the particles, vector:(N*2)
%r_step=10: r interval to sample on $ln(p)\sim r^2$
%r_num=10:The number of r\_step where we calculate the P(r).
%---------------------------------------------------------------------------
%Output:
%prob: prob distribution of P(r), vector:1*r_num
%---------------------------------------------------------------------------

N=length(loc);
loc=loc.^2;
loc=((sqrt(loc(:,1)+loc(:,2))));
prob=zeros(1,r_num);
for i=1:N
    index=floor(loc(i)/r_step)+1;
    if index<=r_num
        prob(index)=prob(index)+1;
    end
end


% pt=1;
% for i=1:N
%     if loc(i)<=(pt)*r_step-dr/2
%         continue
%     end
%     if (pt)*r_step-dr/2<loc(i) & loc(i)<=(pt)*r_step+dr/2
%         prob(pt)=prob(pt)+1;
%     end
%     if loc(i)>(pt)*r_step+dr/2
%         if pt>=r_num
%             break
%         end
%         if (pt+1)*r_step-dr/2<loc(i) & loc(i)<=(pt+1)*r_step+dr/2
%             prob(pt+1)=prob(pt+1)+1;
%         end
%         pt=pt+1;
%         continue
%     end
% end
prob=prob/(N*r_step*2*pi);
for i=1:r_num
    prob(i)=prob(i)/((i-0.5)*r_step);
end
end