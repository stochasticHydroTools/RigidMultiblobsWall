close all
clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
E2E = dlmread('end_to_end_distances.txt');
[ndata,nfib] = size(E2E);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
l_p = 4;
Nlk = 10;
L = 2.0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DATA = load(['./MCMC_data/MCMC_alpha_stat_' num2str(l_p) '_N_' num2str(Nlk) '.mat']);
r = DATA.b;
dist = DATA.m_rd;
%%%%%%%%%%%%%%%%%%
plot(r,dist,'k-','linewidth',3)
hold all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbins = length(r);

Hists = NaN(nfib,nbins);
for fib = 1:nfib
    [h,b] = hist(E2E(:,fib)./L,r);
    Hists(fib,:) = h./trapz(b,h);
end
eb = errorbar(b,mean(Hists),(2.0./sqrt(nfib))*std(Hists))
set(eb,'linewidth',2)
set(gca,'linewidth',3,'fontsize',25,'TickLabelInterpreter','latex')
grid on
xlim([0.8 1.0])
ylim([0 1.2*max(mean(Hists))])