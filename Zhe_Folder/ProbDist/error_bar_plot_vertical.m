%Author: Zhe Chen
%Plot the error bar
function error_bar_plot_vertical(prob,r_step,sample,dt,Sigma)
%Plot the error bar in the vertical direction
%---------------------------------
%Input
%prob: probability of particles
%r_step:r interval to sample on $ln(p)\sim r^2$
%sample:Prob Distribution will be recorded at every 'sample' time steps.
%dt=1:Time step $\Delta t$
%Sigma:covatiance matrix
%---------------------------------
close all
set(0,'defaulttextinterpreter','latex')
set(0,'defaultAxesFontName', 'Times New Roman')
set(0,'DefaultAxesFontSize',15)
[repeat,r_num,t_num]=size(prob);
r_num=r_num/2;
sigma_rec=zeros(t_num,3);
Sigma=mean(Sigma,'omitnan');
for t=2:t_num
    temp0=squeeze(prob(:,:,t));
    temp=log(temp0);
    p0=mean(temp0);
    p=mean(temp);
    err0=2*std(temp0)/sqrt(repeat);
    r2=((linspace(1,r_num,r_num)-0.5)*r_step).^2;
    r2=[-fliplr(r2),r2];
    %     temp=polyfit(r2(3:end),p(3:end),1);
    temp1=polyfit(r2(1:r_num),p(1:r_num),1);
    [~,~,~,~,stats1] = regress(p(1:r_num)',[r2(1:r_num)',ones(r_num,1)]);
    temp2=polyfit(r2(r_num+1:end),p(r_num+1:end),1);
    [~,~,~,~,stats2] = regress(p(r_num+1:end)',[r2(r_num+1:end)',ones(r_num,1)]);
    figure1 = figure;
    set(gcf,'visible','off')
    set(gca,'YScale','log')
    set(gcf,'Position',[0,0,850,600])
    hold on
    box on
    % Create errorbar
    errorbar(r2,p0,err0,'+','DisplayName','Simulation Data');
    % Create plot
    plot(r2(1:r_num),exp(polyval(temp1,r2(1:r_num))),'DisplayName','Regress below the plane');
    plot(r2(1+r_num:end),exp(polyval(temp2,r2(1+r_num:end))),'DisplayName','Regress above the plane');
    %     fplot(@(r2) exp(-sign(r2)*r2/(2*Sigma(t)))/sqrt((2*pi*Sigma(t))),[r2(1),r2(end)],'DisplayName','Estimated');
    plot(r2,exp(-sign(r2).*r2/(2*Sigma(t)))/sqrt((2*pi*Sigma(t))),'DisplayName','Estimated');
    % Create xlabel
    z=xlabel({'$sign(r)r^2$, where r=h-H'});
    set(z,'Interpreter','latex');
    % Create title
    z=title({['t=',num2str((t-1)*sample*dt),';  $\sigma_{left}=$',num2str(sqrt(1/2/temp1(1))),';  $\sigma_{right}=$',num2str(sqrt(-1/2/temp2(1))),'; $\sigma_{estimate}=$',num2str(sqrt(Sigma(t)))]});
    set(z,'Interpreter','latex');
    % Create ylabel
    z=ylabel({['$P(r,t=$',num2str((t-1)*sample*dt),')']});
    set(z,'Interpreter','latex');
    
    annotation(figure1,'textbox',...
        [0.7 0.5 0.3 0.2],...
        'String',{['Goodness of fit:',char(10),'R^2:',num2str(stats2(1))]},...
        'FitBoxToText','off',...
        'EdgeColor',[1 1 1]);
    
    % Create textbox
    annotation(figure1,'textbox',...
        [0.2 0.5 0.3 0.2],...
        'String',{['Goodness of fit:', char(10),'R^2:',num2str(stats1(1))]},...
        'FitBoxToText','off',...
        'EdgeColor',[1 1 1]);
    legend(gca,'show')    
    hold off
    sigma_rec(t,:)=[sqrt(1/2/temp1(1)),sqrt(-1/2/temp2(1)),sqrt(Sigma(t))];
    saveas(figure1,['fig/vertical/t=',num2str((t-1)*sample*dt),'.eps'],'epsc');
end
% rec=[sigma_rec,sqrt(2*(linspace(0,(t_num-1)*sample*dt,t_num)+sigma0^2/2/D))'];
save('sigma_vertical.mat','sigma_rec');

end