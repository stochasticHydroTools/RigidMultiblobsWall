%Author: Zhe Chen
%Plot the error bar
function error_bar_plot(prob,r_step,sample,dt,sigma0,D)
%Plot the error bar
%---------------------------------
%Input
%prob: probability of particles
%r_step:r interval to sample on $ln(p)\sim r^2$
%sample:Prob Distribution will be recorded at every 'sample' time steps.
%dt=1:Time step $\Delta t$
%sigma0:The initial sigma
%---------------------------------

[repeat,r_num,t_num]=size(prob);
sigma_rec=zeros(t_num,1);
for t=1:t_num
    temp=log(squeeze(prob(:,:,t)));
    p=mean(temp);
    err=2*std(temp)/sqrt(repeat);
    r2=((linspace(1,r_num,r_num)-0.5)*r_step).^2;
%     temp=polyfit(r2(3:end),p(3:end),1);
    temp=polyfit(r2,p,1);
    figure1 = figure;
    set(gcf,'visible','off')
    hold on
    % Create errorbar
    errorbar(r2,p,err,'+');
    % Create plot
    plot(r2,polyval(temp,r2));
    % Create xlabel
    xlabel({'r^2'});
    % Create title
    title({['t=',num2str((t-1)*sample*dt),';sigma=',num2str(sqrt(-1/2/temp(1)))]});
    % Create ylabel
    ylabel({['log(P(r,t=',num2str((t-1)*sample*dt),'))']});
%     annotation(figure1,'textbox',...
%         [0.641655105973025 0.633333333333333 0.0394624277456649 0.0385964912280703],...
%         'String',{['sigma=',num2str(sqrt(-1/2/temp(1)))]},...
%         'FitBoxToText','off',...
%         'EdgeColor','none');
    hold off  
    sigma_rec(t)=sqrt(-1/2/temp(1));
    
    saveas(figure1,['t=',num2str((t-1)*sample*dt),'.png']);
end
rec=[sigma_rec,sqrt(2*(linspace(0,(t_num-1)*sample*dt,t_num)+sigma0^2/2/D))'];
save('sigma.mat','rec')

figure1 = figure;
% set(gcf,'visible','off')
axes1 = axes('Parent',figure1);
hold(axes1,'on')
fplot(@(t)sqrt(2*D*(t+sigma0^2/2/D)), [0,(t_num-1)*sample*dt],'DisplayName','Analysis',...
    'Color',[0 0 1]);
scatter((linspace(0,(t_num-1)*sample*dt,t_num))',sigma_rec,20,'filled','DisplayName','Simulation','MarkerFaceColor','flat',...
    'MarkerEdgeColor','none');
xlabel({'t'});
title({'\sigma vs t'});
ylabel({'\sigma'});
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.561607142857143 0.826190476190476 0.178571428571429 0.0773809523809524]);
hold off
saveas(figure1,'sigma.png');
end