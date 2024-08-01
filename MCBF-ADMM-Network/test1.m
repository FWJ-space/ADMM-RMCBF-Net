% 
% load('D:\MCBF\2_2_8_testData_X_200.mat','data_X')
% 
% 
% for i=1:200
%     for j=1:8
%         for k=1:8
%             data_r(i,k,j)=data_X((2*k-1)+16*(j-1),i);
%             data_i(i,k,j)=data_X((2*k)+16*(j-1),i);
%         end
%         
%     end
% end
% 
% save('D:\Python36\data_r.mat','data_r')
% save('D:\Python36\data_i.mat','data_i')


% E1=[0 0 1 0;
%     0 0 0 1;
%     1 0 0 0;
%     0 1 0 0];%ÏßĞÔÓ³Éä¾ØÕó
% E2=[1 0 0 0;
%     0 1 0 0;
%     0 0 1 0;
%     0 0 0 1];
% 
% E=[E1.' E2.'].';
% E3=pinv(E)

% 
% data_i(2,:,:)=data_i(3,:,:);
% data_i(4,:,:)=data_i(3,:,:);
% data_i(5,:,:)=data_i(6,:,:);
% data_i(11,:,:)=data_i(12,:,:);
% data_i(22,:,:)=data_i(23,:,:);
% data_i(30,:,:)=data_i(31,:,:);
% data_i(35,:,:)=data_i(36,:,:);
% data_i(39,:,:)=data_i(40,:,:);
% data_i(48,:,:)=data_i(49,:,:);
% 
% 
% 
% 
% 
% data_r(2,:,:)=data_r(3,:,:);
% data_r(4,:,:)=data_r(3,:,:);
% data_r(5,:,:)=data_r(6,:,:);
% data_r(11,:,:)=data_r(12,:,:);
% data_r(22,:,:)=data_r(23,:,:);
% data_r(30,:,:)=data_r(31,:,:);
% data_r(35,:,:)=data_r(36,:,:);
% data_r(39,:,:)=data_r(40,:,:);
% data_r(48,:,:)=data_r(49,:,:);
% 
% 
% P_out=P_out';
% 
% P_out(2,:)=P_out(3,:);
% P_out(4,:)=P_out(3,:);
% P_out(5,:)=P_out(6,:);
% P_out(11,:)=P_out(12,:);
% P_out(22,:)=P_out(23,:);
% P_out(30,:)=P_out(31,:);
% P_out(35,:)=P_out(36,:);
% P_out(39,:)=P_out(40,:);
% P_out(48,:)=P_out(49,:);
% P_out=P_out';

% data_X = data_X';
% data_X(2,:)=data_X(3,:);
% data_X(4,:)=data_X(3,:);
% data_X(5,:)=data_X(6,:);
% data_X(11,:)=data_X(12,:);
% data_X(22,:)=data_X(23,:);
% data_X(30,:)=data_X(31,:);
% data_X(35,:)=data_X(36,:);
% data_X(39,:)=data_X(40,:);
% data_X(48,:)=data_X(49,:);
% data_X = data_X';


% load('D:\Python36\matlab.mat')


% for i=1:7
%     for j=1:70
%         for k=1:100
%             P_final_out1(j,k,i)= P_final_sum(i,j,k);
%         end
%         
%     end
% end


P_final1=P_final_sum(:,:,1);
P_final2=P_final_sum(:,:,2);
P_final3=P_final_sum(:,:,3);
% P_final4=P_final_sum(:,:,4);
% P_final5=P_final_sum(:,:,5);
% P_final6=P_final_sum(:,:,6);


