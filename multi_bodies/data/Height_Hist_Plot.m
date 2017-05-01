clc
clear all
close all

a = 0.656;


for k = 1:2
    switch k
        case 1
            path = './BlobTest/tol_1e4/sc'
            N = 8;
            bods = 25;
        case 2
            path = './BlobTest/tol_1e3_dt_0.64/sc'
            N = 8;
            bods = 25;
    end
        
    C = [];

    bins = 200;
    b = linspace(0,6,bins);
    H = NaN*ones(N,bins);
    for i =1:N
        i
        B = dlmread([path num2str(i) '.blob_25_phi_0.25.config']);
        remove = 1:(bods+1):length(B);
        Ct = B(:,3);
        Ct(remove) = [];
        [h,b] = hist(Ct,b);
        H(i,:) = a*h./trapz(b,h);
    end
    dist = mean(H);
    er = std(H)./sqrt(N);
    figure(2)
    errorbar(b/a,dist,abs(er),'-o')
    hold all
    title('histogram of heights')
    set(gca,'XTick',[0:6])
    set(gca,'YTick',[0:0.125:1.25])
    axis([0 6 0 1.25])
    
end
hold off