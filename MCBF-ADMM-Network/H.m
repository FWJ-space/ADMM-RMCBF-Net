
function H = H(N,Nt)

for j=1:N 
   j;
   h_head=sqrt(0.5)*randn(Nt,1)+1i*sqrt(0.5)*randn(Nt,1);   %�������Ӹ���˹�����Ԥ�����ŵ�����
   H(:,j)=h_head;%�ŵ�˥��*1.739*10^-4,�����ɵ��ŵ��ŵ��ŵ�������
end  
