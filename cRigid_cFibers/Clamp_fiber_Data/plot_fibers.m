% A = dlmread('Obj_Test_Det_Twirl_fibers_100x100_N100_data0.txt');
% for k = 1:100
%     fib = (k-1);
%     x = A(3*fib+1,:);
%     y = A(3*fib+2,:);
%     z = A(3*fib+3,:);
%     plot3(x,y,z,'o')
%     hold all
% end
% daspect([1 1 1])

% for k = 0:399
% A = dlmread(['Obj_Test_Det_Twirl_fibers_100x100_N100_data' num2str(k) '.txt']);
% x = A(:,1);
% y = A(:,2);
% z = A(:,3);
% plot3(x,y,z,'o')
% daspect([1 1 1])
% zlim([0 max(z)+1])
% xlim([0 6])
% ylim([0 6])
% view([0 0])
% drawnow
% end


for k = 1
A = dlmread(['test_data' num2str(k) '.txt']);
[blobs,Fibx3] = size(A)
for j = 1:(Fibx3/3)
x = A(:,3*(j-1)+1);
y = A(:,3*(j-1)+2);
z = A(:,3*(j-1)+3);
plot3(x,y,z,'o')
daspect([1 1 1])
zlim([0 3])
xlim([-2 2])
ylim([-1 3])
view([0 90])
hold all
end
drawnow
hold off
end