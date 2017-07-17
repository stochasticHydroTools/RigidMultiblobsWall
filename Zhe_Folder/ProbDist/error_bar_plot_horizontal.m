%Author: Zhe Chen
%Plot the error bar
function error_bar_plot_horizontal(prob,r_step,sample,dt,hydro_interaction,Sigma,sigma0,D,plot_no_log)
%Plot the error bar in the horizontal direction
%---------------------------------
%Input
%prob: probability of particles
%r_step:r interval to sample on $ln(p)\sim r^2$
%sample:Prob Distribution will be recorded at every 'sample' time steps.
%dt=1:Time step $\Delta t$
%hydro_interaction: with or without HI
%Sigma: covariance matrix
%sigma0: initial sigma of gaussion distribution in the parallel direction
%D: diffusion coefficient
%plot_no_log: whether plot prob distribution without log'
%---------------------------------
close all
set(0,'defaulttextinterpreter','latex')
set(0,'defaultAxesFontName', 'Times New Roman')
set(0,'DefaultAxesFontSize',12)
if hydro_interaction
    [repeat,r_num,t_num]=size(prob);
    sigma_rec=zeros(t_num);
    Sigma=squeeze(mean(Sigma,'omitnan'));
    for t=1:t_num
        temp0=squeeze(prob(:,:,t));
        temp=log(temp0);
        p0=mean(temp0);
        p=mean(temp);
        err0=2*std(temp0)/sqrt(repeat);
        err=2*std(temp)/sqrt(repeat);
        r=(linspace(1,r_num,r_num)-0.5)*r_step;
        r2=r.^2;
        %     r2=[-fliplr(r2),r2];
        %     temp=polyfit(r2(3:end),p(3:end),1);
%         temp1=polyfit(r2(end-5:end),p(end-5:end),1);
%         [~,~,~,~,stats1] = regress(p(end-5:end)',[r2(end-5:end)',ones(6,1)]);
        %     temp2=polyfit(r2(r_num+1:end),p(r_num+1:end),1);
        %     [~,~,~,~,stats2] = regress(p(r_num+1:end)',[r2(r_num+1:end)',ones(r_num,1)]);
        figure1 = figure;
        set(gcf,'visible','off')
        hold on
        % Create errorbar
        errorbar(r2,p,err,'+','DisplayName','Simulation Data');
        % Create plot
        %     plot(r2(1:r_num),polyval(temp1,r2(1:r_num)));
        Sigma_temp=mean(diag(Sigma(:,:,t)));
        x=linspace(r2(1),r2(end),100);
        plot(x,-x/(2*Sigma_temp)-log(2*pi*Sigma_temp),'DisplayName','Estimated Gaussian Distribution');
        %     plot(r2(1+r_num:end),polyval(temp2,r2(1+r_num:end)));
        % Create xlabel
        xlabel({'$r^2$'});
        % Create title
        title({['t=',num2str((t-1)*sample*dt),';  $\sigma_{estimate}=$',num2str(sqrt(Sigma_temp)),';With HI']});
        % Create ylabel
        ylabel({['log(P(r,t=',num2str((t-1)*sample*dt),'))']});
        legend(gca,'show')
        
%         annotation(figure1,'textbox',...
%             [0.7 0.7 0.3 0.2],...
%             'String',{['Goodness of fit:',char(10),'R^2:',num2str(stats1(1))]},...
%             'FitBoxToText','off',...
%             'EdgeColor',[1 1 1]);
%         
        %     % Create textbox
        %     annotation(figure1,'textbox',...
        %         [0.2 0.7 0.3 0.2],...
        %         'String',{['Goodness of fit:', char(10),'R^2:',num2str(stats1(1))]},...
        %         'FitBoxToText','off',...
        %         'EdgeColor',[1 1 1]);
        
        hold off
        sigma_rec(t)=[sqrt(Sigma_temp)];
        saveas(figure1,['fig/hori/t=',num2str((t-1)*sample*dt),'.eps'],'epsc');

        
        
        if plot_no_log
            
            figure0 = figure;
            set(gcf,'visible','off')
            hold on
            % Create errorbar
            errorbar(r,p0,err0,'+','DisplayName','Simulation Data');
            x=linspace(r(1),r(end),100);
            plot(x,exp(-x.^2/(2*Sigma_temp))./(2*pi*Sigma_temp),'DisplayName','Estimated Gaussian Distribution');
            %     plot(r2(1+r_num:end),polyval(temp2,r2(1+r_num:end)));
            % Create xlabel
            xlabel({'$r$'});
            % Create title
            title({['t=',num2str((t-1)*sample*dt),';  $\sigma_{estimate}=$',num2str(sqrt(Sigma_temp)),';With HI']});
            % Create ylabel
            ylabel({['P(r,t=',num2str((t-1)*sample*dt),')']});
            legend(gca,'show')
            hold off
            saveas(figure0,['fig/hori/no_log_t=',num2str((t-1)*sample*dt),'.eps'],'epsc');
        end
    end
    % rec=[sigma_rec,sqrt(2*(linspace(0,(t_num-1)*sample*dt,t_num)+sigma0^2/2/D))'];
    save('sigma.mat','sigma_rec')
    %

    
    
    
else
    [repeat,r_num,t_num]=size(prob);
    sigma_rec=zeros(t_num,1);
    Sigma=squeeze(mean(Sigma,'omitnan'));
    for t=1:t_num
        temp=log(squeeze(prob(:,:,t)));
        p=mean(temp);
        err=2*std(temp)/sqrt(repeat);
        r2=((linspace(1,r_num,r_num)-0.5)*r_step).^2;
        %     r2=[-fliplr(r2),r2];
        %     temp=polyfit(r2(3:end),p(3:end),1);
        temp1=polyfit(r2,p,1);
        [~,~,~,~,stats1] = regress(p',[r2',ones(r_num,1)]);
        %     temp2=polyfit(r2(r_num+1:end),p(r_num+1:end),1);
        %     [~,~,~,~,stats2] = regress(p(r_num+1:end)',[r2(r_num+1:end)',ones(r_num,1)]);
        figure1 = figure;
        set(gcf,'visible','off')
        hold on
        % Create errorbar
        errorbar(r2,p,err,'+','DisplayName','Simulation Data');
        % Create plot
        %     plot(r2(1:r_num),polyval(temp1,r2(1:r_num)));
%         Sigma_temp=-1/2/temp1(1);
        Sigma_temp=mean(diag(Sigma(:,:,t)));
        Sigma_analysis=(2*D*((t-1)*sample*dt+sigma0^2/2/D));
        x=linspace(r2(1),r2(end),100);
        plot(x,-x/(2*Sigma_analysis)-log(2*pi*Sigma_analysis),'DisplayName','Analysis');
        plot(x,-x/(2*Sigma_temp)-log(2*pi*Sigma_temp),'DisplayName','Estimated');
        %     plot(r2(1+r_num:end),polyval(temp2,r2(1+r_num:end)));
        % Create xlabel
        xlabel({'$r^2$'});
        % Create title
        title({['t=',num2str((t-1)*sample*dt),';  $\sigma_{Estimated}=$',num2str(sqrt(Sigma_temp)),'; $\sigma_{analysis}=$',num2str(sqrt(Sigma_analysis)),'; Without HI']});
        % Create ylabel
        ylabel({['log(P(r,t=',num2str((t-1)*sample*dt),'))']});
        
%         annotation(figure1,'textbox',...
%             [0.6 0.5 0.3 0.2],...
%             'String',{['Goodness of fit:',char(10),'R^2:',num2str(stats1(1))]},...
%             'FitBoxToText','off',...
%             'EdgeColor',[1 1 1]);
        legend(gca,'show')
        
        %     % Create textbox
        %     annotation(figure1,'textbox',...
        %         [0.2 0.7 0.3 0.2],...
        %         'String',{['Goodness of fit:', char(10),'R^2:',num2str(stats1(1))]},...
        %         'FitBoxToText','off',...
        %         'EdgeColor',[1 1 1]);
        
        hold off
        sigma_rec(t)=sqrt(Sigma_temp);
%         sigma_rec(t)=[sqrt(-1/2/temp1(1))];
        saveas(figure1,['fig/hori/t=',num2str((t-1)*sample*dt),'.eps'],'epsc');
    end
    % rec=[sigma_rec,sqrt(2*(linspace(0,(t_num-1)*sample*dt,t_num)+sigma0^2/2/D))'];
    save('sigma_hori.mat','sigma_rec')
    figure1 = figure;
    set(gcf,'visible','off')
    axes1 = axes('Parent',figure1);
    hold(axes1,'on')
    fplot(@(t)sqrt(2*D*(t+sigma0^2/2/D)),[0,(t_num-1)*sample*dt],'DisplayName','Analysis',...
        'Color',[0 0 1]);
    scatter((linspace(0,(t_num-1)*sample*dt,t_num))',sigma_rec,20,'filled','DisplayName','Simulation','MarkerFaceColor','flat',...
        'MarkerEdgeColor','none');
    xlabel({'t'});
    title({'$\sigma$ vs t'});
    ylabel({'$\sigma$'});
    legend1 = legend(axes1,'show');
    set(legend1,...
        'Position',[0.561607142857143 0.826190476190476 0.178571428571429 0.0773809523809524]);
    hold off
    saveas(figure1,'fig/hori/sigma.eps','epsc');
end
end